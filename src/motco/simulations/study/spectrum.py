"""Per-cell latent configuration-spectrum summaries and eigengap stratification.

The relative eigengap ``(l1 - l2) / sum(l)`` of the centered stage-mean
configuration in the latent space is the observable the geometry audit
(``docs/reports/geometry-audit-2026-09-01.md``) found governs whether an
orientation is resolvable: a near-isotropic configuration has no dominant axis
for PC1 to lock onto, its ``angle`` permutation null is wide, and the replicate
cannot reject however large its observed angle.

Both tables here read **recorded** values only — they never regenerate a dataset
or recompute a spectrum from raw data — and they qualify reporting rather than
changing any statistic or decision.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from motco.simulations.grid import SimulationReplicateResult
from motco.stats.trajectory import pooled_relative_eigengap

#: Modes whose power is stratified by eigengap. Orientation is the mode the
#: eigengap governs; the others move statistics the eigengap does not gate.
STRATIFIED_MODES: tuple[str, ...] = ("orientation",)

#: Phases carrying power cells (as opposed to the Type I controls).
POWER_PHASES: tuple[str, ...] = ("power_primary", "power_ofat")

#: Number of within-cell strata. Terciles: enough to show a monotone trend at
#: 100 replicates per cell without splitting the counts past usefulness.
DEFAULT_N_STRATA = 3

_QUANTILES: tuple[tuple[str, float], ...] = (
    ("q25", 0.25),
    ("median", 0.50),
    ("q75", 0.75),
)


def pooled_eigengap(record: SimulationReplicateResult) -> float | None:
    """Recorded pooled relative eigengap for one replicate, or ``None``.

    ``None`` covers both a record written before the field existed and a
    recorded configuration whose eigengap is undefined; the two are separated by
    :func:`has_spectrum`.
    """

    return pooled_relative_eigengap(record.config_spectrum)


def group_eigengaps(record: SimulationReplicateResult) -> dict[str, float | None]:
    """Recorded per-group relative eigengaps for one replicate, keyed by group."""

    groups = (record.config_spectrum or {}).get("groups") or {}
    return {str(name): _eigengap(entry) for name, entry in dict(groups).items()}


def has_spectrum(record: SimulationReplicateResult) -> bool:
    """Whether the record carries a spectrum block at all."""

    return bool(record.config_spectrum)


def _eigengap(entry: Any) -> float | None:
    if not isinstance(entry, Mapping):
        return None
    value = entry.get("relative_eigengap")
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def summarize_config_spectrum(
    records: Sequence[SimulationReplicateResult],
) -> pd.DataFrame:
    """Per-cell summary of the recorded pooled and per-group eigengaps.

    One row per (cell, configuration), where configuration is ``pooled`` or a
    group label. ``n_recorded`` counts replicates carrying a spectrum block and
    ``n_available`` those whose eigengap is defined, so records predating the
    field and degenerate configurations stay separately visible.
    """

    buckets: dict[tuple[str, str], list[float | None]] = {}
    identity: dict[tuple[str, str], dict[str, Any]] = {}
    recorded: dict[tuple[str, str], int] = {}

    for record in records:
        if record.status != "completed":
            continue
        meta = dict(record.cell_metadata)
        mode = _resolve_mode(meta, record.phase)
        effect_size = meta.get("effect_size")
        entries: list[tuple[str, float | None]] = [("pooled", pooled_eigengap(record))]
        entries.extend(sorted(group_eigengaps(record).items()))
        if not has_spectrum(record):
            # Still counted, so a cell of legacy records reports zeros rather
            # than disappearing from the table.
            entries = [("pooled", None)]
        for configuration, value in entries:
            key = (record.cell_id, configuration)
            buckets.setdefault(key, []).append(value)
            recorded[key] = recorded.get(key, 0) + int(has_spectrum(record))
            identity.setdefault(
                key,
                {
                    "trajectory_mode": mode,
                    "effect_size": None if effect_size is None else float(effect_size),
                    "cell_id": record.cell_id,
                    "phase": record.phase,
                    "configuration": configuration,
                },
            )

    rows: list[dict[str, Any]] = []
    for key, values in buckets.items():
        available = [value for value in values if value is not None]
        row = dict(identity[key])
        row.update(
            {
                "n_replicates": len(values),
                "n_recorded": recorded[key],
                "n_available": len(available),
                "mean_eigengap": _mean(available),
                "sd_eigengap": _sd(available),
            }
        )
        for name, q in _QUANTILES:
            row[f"{name}_eigengap"] = (
                float(np.quantile(available, q)) if available else None
            )
        row["min_eigengap"] = min(available) if available else None
        row["max_eigengap"] = max(available) if available else None
        rows.append(row)

    frame = pd.DataFrame(
        rows,
        columns=[
            "trajectory_mode",
            "effect_size",
            "cell_id",
            "phase",
            "configuration",
            "n_replicates",
            "n_recorded",
            "n_available",
            "mean_eigengap",
            "sd_eigengap",
            "q25_eigengap",
            "median_eigengap",
            "q75_eigengap",
            "min_eigengap",
            "max_eigengap",
        ],
    )
    if frame.empty:
        return frame
    return frame.sort_values(
        ["trajectory_mode", "effect_size", "cell_id", "configuration"]
    ).reset_index(drop=True)


def stratify_power_by_eigengap(
    records: Sequence[SimulationReplicateResult],
    *,
    alpha: float = 0.05,
    statistics: Sequence[str] = ("delta", "angle", "shape"),
    modes: Sequence[str] = STRATIFIED_MODES,
    n_strata: int = DEFAULT_N_STRATA,
) -> pd.DataFrame:
    """Rejection rates within eigengap strata, for the stratified power cells.

    Strata are within-cell quantile bins of the **recorded** pooled eigengap, so
    every replicate in a cell is compared against its own cell's distribution.
    A cell whose records carry no spectrum yields one row per statistic marked
    ``status='unavailable'`` rather than silently vanishing — the covariate is
    missing, which is a different fact from a degenerate geometry.
    """

    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1.")
    if n_strata < 2:
        raise ValueError("n_strata must be at least 2.")

    rows: list[dict[str, Any]] = []
    for cell_id, group in _stratified_cells(records, modes).items():
        meta = dict(group[0].cell_metadata)
        effect_size = meta.get("effect_size")
        identity = {
            "trajectory_mode": _resolve_mode(meta, group[0].phase),
            "effect_size": None if effect_size is None else float(effect_size),
            "cell_id": cell_id,
            "phase": group[0].phase,
        }
        gaps = [(record, pooled_eigengap(record)) for record in group]
        usable = [(record, gap) for record, gap in gaps if gap is not None]
        if not usable:
            for statistic in statistics:
                rows.append(
                    {
                        **identity,
                        "statistic": statistic,
                        "stratum": None,
                        "n_strata": n_strata,
                        "eigengap_low": None,
                        "eigengap_high": None,
                        "mean_eigengap": None,
                        "n_replicates": len(group),
                        "n_available": 0,
                        "n_rejected": 0,
                        "rejection_rate": None,
                        "monte_carlo_se": None,
                        "status": "unavailable",
                    }
                )
            continue

        assignments = _quantile_strata([gap for _, gap in usable], n_strata)
        for stratum in range(n_strata):
            members = [
                (record, gap)
                for (record, gap), index in zip(usable, assignments)
                if index == stratum
            ]
            stratum_gaps = [gap for _, gap in members]
            for statistic in statistics:
                p_values = [
                    float(p)
                    for record, _ in members
                    if (p := record.p_values.get(statistic)) is not None
                    and math.isfinite(float(p))
                ]
                rejected = sum(p < alpha for p in p_values)
                rate = rejected / len(p_values) if p_values else None
                rows.append(
                    {
                        **identity,
                        "statistic": statistic,
                        "stratum": stratum,
                        "n_strata": n_strata,
                        "eigengap_low": min(stratum_gaps) if stratum_gaps else None,
                        "eigengap_high": max(stratum_gaps) if stratum_gaps else None,
                        "mean_eigengap": _mean(stratum_gaps),
                        "n_replicates": len(members),
                        "n_available": len(p_values),
                        "n_rejected": rejected,
                        "rejection_rate": rate,
                        "monte_carlo_se": (
                            math.sqrt(rate * (1.0 - rate) / len(p_values))
                            if rate is not None
                            else None
                        ),
                        "status": "ok",
                    }
                )

    frame = pd.DataFrame(
        rows,
        columns=[
            "trajectory_mode",
            "effect_size",
            "cell_id",
            "phase",
            "statistic",
            "stratum",
            "n_strata",
            "eigengap_low",
            "eigengap_high",
            "mean_eigengap",
            "n_replicates",
            "n_available",
            "n_rejected",
            "rejection_rate",
            "monte_carlo_se",
            "status",
        ],
    )
    if frame.empty:
        return frame
    return frame.sort_values(
        ["trajectory_mode", "effect_size", "cell_id", "statistic", "stratum"],
        na_position="last",
    ).reset_index(drop=True)


def _stratified_cells(
    records: Iterable[SimulationReplicateResult],
    modes: Sequence[str],
) -> dict[str, list[SimulationReplicateResult]]:
    wanted = set(modes)
    groups: dict[str, list[SimulationReplicateResult]] = {}
    for record in records:
        if record.status != "completed" or record.phase not in POWER_PHASES:
            continue
        meta = dict(record.cell_metadata)
        if _resolve_mode(meta, record.phase) not in wanted:
            continue
        groups.setdefault(record.cell_id, []).append(record)
    return dict(sorted(groups.items()))


def _quantile_strata(values: Sequence[float], n_strata: int) -> list[int]:
    """Assign each value to a within-cell quantile stratum.

    Ranks are used rather than raw cut points so ties cannot pile every
    replicate into one bin, and the strata stay as close to equal-sized as the
    data allow.
    """

    order = np.argsort(np.asarray(values, dtype=float), kind="stable")
    ranks = np.empty(len(values), dtype=int)
    ranks[order] = np.arange(len(values))
    return [int(rank * n_strata // max(len(values), 1)) for rank in ranks]


def _resolve_mode(meta: Mapping[str, Any], phase: str) -> str:
    mode = meta.get("trajectory_mode")
    if mode is not None:
        return str(mode)
    return "none" if phase.startswith("type_i") else "unknown"


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _sd(values: Sequence[float]) -> float | None:
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


__all__ = [
    "DEFAULT_N_STRATA",
    "POWER_PHASES",
    "STRATIFIED_MODES",
    "group_eigengaps",
    "has_spectrum",
    "pooled_eigengap",
    "stratify_power_by_eigengap",
    "summarize_config_spectrum",
]
