"""Is the RRPP permutation null pivotal under signal?

A test statistic is *pivotal* when its null distribution does not depend on the
nuisance state of the particular dataset. RRPP p-values are exact under
exchangeability whether or not the statistic is pivotal — non-pivotality does
not break validity, it costs **power**: if a replicate whose latent geometry
inflates the observed statistic inflates its own critical value by the same
amount, the signal never clears its own bar.

This module measures that from study records alone, using the per-replicate
permutation-null summary persisted by
:class:`~motco.simulations.grid.SimulationReplicateResult`. Three views:

1. :func:`association_table` — within a cell, across replicates, how strongly
   the null's location and spread track the observed statistic.
2. :func:`rejection_split_table` — the observed statistic split by rejection
   outcome, which is where the Phase 4 inversion (non-rejecting replicates
   carrying the *larger* mean observed angle) shows up.
3. :func:`standardized_counterfactual_table` — the cross-replicate standardized
   recalibration.

**The counterfactual is a diagnostic, not a deployable test.** Standardizing an
observed statistic by its own null's mean and sd and comparing it to that same
null's quantiles is a strictly monotone transform applied to both sides: the
p-value is unchanged, and within-replicate studentization cannot recover a
single rejection (pinned in the test suite). The counterfactual here instead
calibrates each replicate's ``z`` against the ``z`` distribution of the
*null-control cells*, which borrows a reference distribution that does not exist
in real data. A production pivotal test would need a nested/double permutation
or an analytically studentized statistic; neither is in scope here.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from motco.simulations.grid import SimulationReplicateResult

STATISTICS: tuple[str, ...] = ("delta", "angle", "shape")

#: Null-summary keys the association table regresses the observed statistic
#: against: the null's location, its spread, and the value the test actually
#: compares against at alpha = 0.05.
NULL_TARGETS: tuple[str, ...] = ("mean", "sd", "q95")


class PivotalityError(ValueError):
    """Raised when records cannot support the requested pivotality analysis."""


@dataclass(frozen=True)
class CellKey:
    """Identity of one analyzed cell."""

    cell_id: str
    phase: str
    trajectory_mode: str
    effect_size: float | None

    @property
    def label(self) -> str:
        effect = "na" if self.effect_size is None else f"{self.effect_size:g}"
        return f"{self.trajectory_mode}@{effect}[{self.phase}]"


@dataclass(frozen=True)
class AssociationRow:
    """Association between the observed statistic and one feature of its own null."""

    cell: CellKey
    statistic: str
    null_target: str
    n_replicates: int
    #: Pearson correlation across replicates. ``None`` when it is undefined
    #: (fewer than three usable replicates, or either side is constant).
    correlation: float | None
    #: Approximate 95% interval for ``correlation`` via the Fisher z transform.
    #: This is the "how strongly does the null track the observed value" measure
    #: required by the spec, reported so a weak result cannot be over-read.
    correlation_ci_low: float | None
    correlation_ci_high: float | None
    #: Slope of ``null_target`` on the observed statistic. A pivotal statistic
    #: has slope 0 for ``mean`` and ``q95``; a slope near 1 means the critical
    #: value moves point-for-point with the signal.
    slope: float | None
    mean_observed: float | None
    mean_null_target: float | None


@dataclass(frozen=True)
class RejectionSplitRow:
    """Observed statistic and critical value split by rejection outcome."""

    cell: CellKey
    statistic: str
    alpha: float
    n_rejecting: int
    n_non_rejecting: int
    mean_observed_rejecting: float | None
    mean_observed_non_rejecting: float | None
    mean_critical_rejecting: float | None
    mean_critical_non_rejecting: float | None
    #: ``True`` when non-rejecting replicates carry the larger mean observed
    #: statistic — the Phase 4 inversion.
    inverted: bool | None


@dataclass(frozen=True)
class StandardizedRow:
    """As-specified rejection rate beside the cross-replicate standardized rate."""

    cell: CellKey
    statistic: str
    alpha: float
    n_replicates: int
    as_specified_rate: float | None
    standardized_rate: float | None
    #: The ``z`` cut taken from the pooled null-control replicates. Shared by
    #: every row of one table, and reported so the borrowed reference is visible.
    z_threshold: float | None
    mean_z: float | None
    is_null_control: bool


def cell_key(record: SimulationReplicateResult) -> CellKey:
    """Identify the cell a record belongs to, from its persisted cell metadata."""

    metadata = record.cell_metadata
    mode = metadata.get("trajectory_mode")
    if mode is None:
        # The Type I baseline cell carries no explicit mode: it *is* ``none``.
        mode = "none" if record.phase.startswith("type_i") else "unknown"
    effect = metadata.get("effect_size")
    return CellKey(
        cell_id=record.cell_id,
        phase=record.phase,
        trajectory_mode=str(mode),
        effect_size=None if effect is None else float(effect),
    )


def group_by_cell(
    records: Iterable[SimulationReplicateResult],
) -> dict[CellKey, list[SimulationReplicateResult]]:
    """Group completed records by cell, in a stable order."""

    groups: dict[CellKey, list[SimulationReplicateResult]] = {}
    for record in records:
        if record.status != "completed":
            continue
        groups.setdefault(cell_key(record), []).append(record)
    return dict(sorted(groups.items(), key=lambda item: item[0].label))


def _observed(record: SimulationReplicateResult, statistic: str) -> float | None:
    value = record.pair_statistics.get(statistic)
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _null_feature(record: SimulationReplicateResult, statistic: str, key: str) -> float | None:
    entry = record.null_summary.get(statistic)
    if not entry or key not in entry:
        return None
    value = float(entry[key])
    return value if math.isfinite(value) else None


def _paired(
    records: Sequence[SimulationReplicateResult], statistic: str, null_target: str
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    for record in records:
        observed = _observed(record, statistic)
        target = _null_feature(record, statistic, null_target)
        if observed is None or target is None:
            continue
        xs.append(observed)
        ys.append(target)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _correlation_with_interval(
    x: np.ndarray, y: np.ndarray
) -> tuple[float | None, float | None, float | None]:
    """Pearson r plus a Fisher-z 95% interval, or ``None`` where undefined."""

    n = x.size
    if n < 3 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return None, None, None
    r = float(np.corrcoef(x, y)[0, 1])
    if not math.isfinite(r):
        return None, None, None
    if abs(r) >= 1.0:
        # A perfect correlation has zero Fisher-z spread; report it without an
        # interval rather than emitting an infinite one.
        return r, None, None
    z = math.atanh(r)
    half_width = 1.959963984540054 / math.sqrt(n - 3)
    return r, math.tanh(z - half_width), math.tanh(z + half_width)


def _slope(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2:
        return None
    variance = float(np.var(x))
    if variance == 0.0:
        return None
    return float(np.cov(x, y, ddof=0)[0, 1] / variance)


def association_table(
    records: Iterable[SimulationReplicateResult],
    *,
    statistics: Sequence[str] = STATISTICS,
    null_targets: Sequence[str] = NULL_TARGETS,
) -> list[AssociationRow]:
    """Per cell and statistic, how strongly the null tracks the observed value.

    A pivotal statistic shows no association: the null mean, spread, and
    critical value are properties of the design, not of the replicate's realized
    signal. A correlation near +1 with a slope near 1 on ``q95`` is the
    non-pivotal case that costs power.
    """

    rows: list[AssociationRow] = []
    for cell, group in group_by_cell(records).items():
        for statistic in statistics:
            for target in null_targets:
                x, y = _paired(group, statistic, target)
                r, low, high = _correlation_with_interval(x, y)
                rows.append(
                    AssociationRow(
                        cell=cell,
                        statistic=statistic,
                        null_target=target,
                        n_replicates=int(x.size),
                        correlation=r,
                        correlation_ci_low=low,
                        correlation_ci_high=high,
                        slope=_slope(x, y),
                        mean_observed=float(x.mean()) if x.size else None,
                        mean_null_target=float(y.mean()) if y.size else None,
                    )
                )
    return rows


def rejection_split_table(
    records: Iterable[SimulationReplicateResult],
    *,
    alpha: float = 0.05,
    statistics: Sequence[str] = STATISTICS,
) -> list[RejectionSplitRow]:
    """Split the observed statistic — and its critical value — by rejection outcome.

    ``inverted`` flags the Phase 4 pattern: non-rejecting replicates carrying a
    *larger* mean observed statistic than rejecting ones. Reporting the mean
    critical value beside it answers the follow-up the spec requires — whether
    replicates with a larger observed statistic also carry a proportionally
    larger bar.
    """

    if not (0 < alpha < 1):
        raise PivotalityError("alpha must be between 0 and 1.")
    rows: list[RejectionSplitRow] = []
    for cell, group in group_by_cell(records).items():
        for statistic in statistics:
            rejecting: list[tuple[float, float | None]] = []
            non_rejecting: list[tuple[float, float | None]] = []
            for record in group:
                p_value = record.p_values.get(statistic)
                observed = _observed(record, statistic)
                if p_value is None or observed is None or not math.isfinite(float(p_value)):
                    continue
                critical = _null_feature(record, statistic, "q95")
                target = rejecting if float(p_value) < alpha else non_rejecting
                target.append((observed, critical))
            mean_obs_rej = _mean_first(rejecting)
            mean_obs_non = _mean_first(non_rejecting)
            rows.append(
                RejectionSplitRow(
                    cell=cell,
                    statistic=statistic,
                    alpha=alpha,
                    n_rejecting=len(rejecting),
                    n_non_rejecting=len(non_rejecting),
                    mean_observed_rejecting=mean_obs_rej,
                    mean_observed_non_rejecting=mean_obs_non,
                    mean_critical_rejecting=_mean_second(rejecting),
                    mean_critical_non_rejecting=_mean_second(non_rejecting),
                    inverted=(
                        None
                        if mean_obs_rej is None or mean_obs_non is None
                        else mean_obs_non > mean_obs_rej
                    ),
                )
            )
    return rows


def _mean_first(pairs: Sequence[tuple[float, float | None]]) -> float | None:
    return float(np.mean([value for value, _ in pairs])) if pairs else None


def _mean_second(pairs: Sequence[tuple[float, float | None]]) -> float | None:
    values = [second for _, second in pairs if second is not None]
    return float(np.mean(values)) if values else None


def replicate_z(record: SimulationReplicateResult, statistic: str) -> float | None:
    """``(observed − null_mean) / null_sd`` for one replicate, or ``None``."""

    observed = _observed(record, statistic)
    mean = _null_feature(record, statistic, "mean")
    sd = _null_feature(record, statistic, "sd")
    if observed is None or mean is None or sd is None or sd <= 0.0:
        return None
    z = (observed - mean) / sd
    return z if math.isfinite(z) else None


def standardized_counterfactual_table(
    records: Iterable[SimulationReplicateResult],
    *,
    alpha: float = 0.05,
    statistics: Sequence[str] = STATISTICS,
    null_control_modes: Sequence[str] = ("none", "translation"),
) -> list[StandardizedRow]:
    """Rejection rate under a ``z`` calibrated on the null-control cells.

    Per replicate, ``z = (observed − null_mean) / null_sd``. The threshold is the
    ``1 − alpha`` quantile of ``z`` pooled over the *null-control* replicates for
    that statistic, and every cell — controls included — is then scored against
    it. Reporting the controls' own rates is what makes a power gain bought with
    an inflated Type I rate visible.

    This borrows a reference distribution that does not exist outside a
    simulation. It is a diagnostic, never a deployable test; see the module
    docstring.
    """

    if not (0 < alpha < 1):
        raise PivotalityError("alpha must be between 0 and 1.")
    groups = group_by_cell(records)
    control_modes = set(null_control_modes)

    thresholds: dict[str, float | None] = {}
    for statistic in statistics:
        pooled = [
            z
            for cell, group in groups.items()
            if cell.trajectory_mode in control_modes
            for record in group
            if (z := replicate_z(record, statistic)) is not None
        ]
        thresholds[statistic] = float(np.quantile(pooled, 1.0 - alpha)) if pooled else None

    rows: list[StandardizedRow] = []
    for cell, group in groups.items():
        for statistic in statistics:
            zs = [z for record in group if (z := replicate_z(record, statistic)) is not None]
            p_values = [
                float(p)
                for record in group
                if (p := record.p_values.get(statistic)) is not None and math.isfinite(float(p))
            ]
            threshold = thresholds[statistic]
            rows.append(
                StandardizedRow(
                    cell=cell,
                    statistic=statistic,
                    alpha=alpha,
                    n_replicates=len(zs),
                    as_specified_rate=(
                        float(np.mean([p < alpha for p in p_values])) if p_values else None
                    ),
                    standardized_rate=(
                        float(np.mean([z >= threshold for z in zs]))
                        if zs and threshold is not None
                        else None
                    ),
                    z_threshold=threshold,
                    mean_z=float(np.mean(zs)) if zs else None,
                    is_null_control=cell.trajectory_mode in control_modes,
                )
            )
    return rows


def rows_to_records(rows: Sequence[object]) -> list[dict[str, object]]:
    """Flatten any of the table row dataclasses into CSV-ready dicts."""

    out: list[dict[str, object]] = []
    for row in rows:
        flat: dict[str, object] = {}
        for name, value in vars(row).items():
            if isinstance(value, CellKey):
                flat.update(
                    {
                        "cell_id": value.cell_id,
                        "phase": value.phase,
                        "trajectory_mode": value.trajectory_mode,
                        "effect_size": value.effect_size,
                    }
                )
            else:
                flat[name] = value
        out.append(flat)
    return out


def write_pivotality_tables(
    records: Sequence[SimulationReplicateResult],
    out_dir: Path,
    *,
    alpha: float = 0.05,
    statistics: Sequence[str] = STATISTICS,
) -> dict[str, Path]:
    """Write the association, rejection-split, and counterfactual tables as CSV."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables: Mapping[str, list[dict[str, object]]] = {
        "pivotality_association": rows_to_records(
            association_table(records, statistics=statistics)
        ),
        "pivotality_rejection_split": rows_to_records(
            rejection_split_table(records, alpha=alpha, statistics=statistics)
        ),
        "pivotality_standardized": rows_to_records(
            standardized_counterfactual_table(records, alpha=alpha, statistics=statistics)
        ),
    }
    paths: dict[str, Path] = {}
    for name, table in tables.items():
        path = out_dir / f"{name}.csv"
        if not table:
            path.write_text("", encoding="utf-8")
            paths[name] = path
            continue
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
        paths[name] = path
    return paths
