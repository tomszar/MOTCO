"""Phase 4 summaries, localization, and the predeclared Phase 5 gate.

Two rules shape this module.

**Thresholds are declared, not embedded.** Every mandatory gate number — alpha,
the tolerance multipliers, the power floor, the confirmation threshold, and which
mode/statistic pairs are mandatory versus descriptive — is read from the study
configuration's :class:`~motco.simulations.study.config.Phase4GateConfig`. No
gate threshold is hard-coded here, so a gate can be re-specified by editing the
committed config rather than this package.

**Construction interpretation is separate from statistical gates.** Off-diagonal
rejection is localized against realized geometry and reported descriptively;
only the predeclared mandatory rules decide ``proceed`` / ``hold`` /
``indeterminate``. Raw distances are never compared across the standardized
feature space and the PLS latent space — each checkpoint is compared against its
*own* zero-effect null instead.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from motco.simulations.grid import SimulationReplicateResult, SimulationSummaryResult
from motco.simulations.study.config import GateRule, Phase4GateConfig

#: Checkpoint order from construction through preprocessing to projection.
CHECKPOINT_ORDER: tuple[str, ...] = (
    "population_native",
    "population_standardized",
    "observed_standardized",
    "pls_latent",
)

#: Measurement space each checkpoint is expressed in. Distances are only ever
#: compared *within* one of these labels.
MEASUREMENT_SPACES: dict[str, str] = {
    "population_native": "native_omic_units",
    "population_standardized": "standardized_feature",
    "observed_standardized": "standardized_feature",
    "pls_latent": "pls_latent",
}

#: How the first checkpoint carrying material off-diagonal geometry is read.
#: These labels describe *localization*, not causality.
CHECKPOINT_CLASSIFICATION: dict[str, str] = {
    "population_native": "construction_present",
    "population_standardized": "construction_present",
    "observed_standardized": "sampling_or_preprocessing_associated",
    "pls_latent": "projection_associated",
}

#: The statistic each trajectory mode is constructed to move.
DIAGONAL_STATISTIC: dict[str, str] = {
    "magnitude": "delta",
    "orientation": "angle",
    "shape": "shape",
    "translation": "delta",
}

_STATISTICS: tuple[str, ...] = ("delta", "angle", "shape")

#: The orientation estimator reports an angle in degrees on ``[0, 180]``.
_MAX_ANGLE_DEGREES = 180.0
_COMPONENTS: tuple[str, ...] = ("observed", "pls_captured", "residual")

#: Descriptive-only materiality threshold for localization, on the scale-free
#: statistic described in :func:`localize_off_diagonal`: a checkpoint counts as
#: material when its normalized value exceeds the zero-effect null at that same
#: checkpoint by at least this much. It gates nothing.
DEFAULT_MATERIALITY_THRESHOLD = 0.05


class Phase4SummaryError(ValueError):
    """Raised when Phase 4 summaries cannot be built from the supplied records."""


# --------------------------------------------------------------------------- #
# Operating characteristics keyed by mode and effect
# --------------------------------------------------------------------------- #


def build_operating_frame(
    summaries: Sequence[SimulationSummaryResult],
    records: Sequence[SimulationReplicateResult],
) -> pd.DataFrame:
    """One row per (mode, effect, statistic) over primary power cells.

    The shared zero-effect anchor is expanded so every mode's power curve
    resolves its ``0.00`` point from that one cell rather than from a per-mode
    duplicate. Anchor-derived rows are flagged so a reader can see that the
    modes share one null and are not independent evidence at that point.
    """

    cell_meta = _cell_metadata_index(records)
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        meta = cell_meta.get(summary.cell_id)
        if meta is None or meta.get("varied_axis") is not None:
            continue
        if summary.phase != "power_primary":
            continue
        effect_size = meta.get("effect_size")
        if effect_size is None:
            continue
        modes = (
            [str(mode) for mode in meta.get("resolves_modes", [])]
            if meta.get("zero_effect_anchor")
            else [str(meta.get("trajectory_mode", "none"))]
        )
        for mode in modes:
            rows.append(
                {
                    "trajectory_mode": mode,
                    "effect_size": float(effect_size),
                    "statistic": summary.statistic,
                    "cell_id": summary.cell_id,
                    "phase": summary.phase,
                    "from_shared_anchor": bool(meta.get("zero_effect_anchor", False)),
                    "rejection_rate": summary.rejection_rate,
                    "monte_carlo_se": summary.monte_carlo_se,
                    "available_replicates": summary.available_replicates,
                    "completed_replicates": summary.completed_replicates,
                }
            )
    frame = pd.DataFrame(
        rows,
        columns=[
            "trajectory_mode",
            "effect_size",
            "statistic",
            "cell_id",
            "phase",
            "from_shared_anchor",
            "rejection_rate",
            "monte_carlo_se",
            "available_replicates",
            "completed_replicates",
        ],
    )
    if frame.empty:
        return frame
    return frame.sort_values(["trajectory_mode", "statistic", "effect_size"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 5.1 realized geometry
# --------------------------------------------------------------------------- #


def summarize_realized_geometry(records: Sequence[SimulationReplicateResult]) -> pd.DataFrame:
    """Long-form geometry by mode, effect, checkpoint, scope, and statistic.

    Path lengths are summarized alongside ``delta`` / ``angle`` / ``shape``.
    Unavailable values (a null path with no direction, or ``shape`` with fewer
    than three stages) are counted rather than coerced to zero, and each row
    carries the measurement space its numbers live in so feature-space and
    latent-space magnitudes are never silently compared.
    """

    buckets: dict[tuple, list[float | None]] = {}
    identity: dict[tuple, dict[str, Any]] = {}
    for record in records:
        if record.status != "completed":
            continue
        geometry = record.realized_geometry or {}
        meta = dict(record.cell_metadata)
        mode = str(meta.get("trajectory_mode") or geometry.get("requested", {}).get("trajectory_mode") or "none")
        effect_size = meta.get("effect_size")
        effect_size = float(effect_size) if effect_size is not None else None
        for checkpoint, scopes in (geometry.get("checkpoints") or {}).items():
            for scope, values in scopes.items():
                for statistic, value in _geometry_statistics(values):
                    key = (mode, effect_size, record.cell_id, checkpoint, scope, statistic)
                    buckets.setdefault(key, []).append(value)
                    identity.setdefault(
                        key,
                        {
                            "trajectory_mode": mode,
                            "effect_size": effect_size,
                            "cell_id": record.cell_id,
                            "phase": record.phase,
                            "checkpoint": checkpoint,
                            "scope": scope,
                            "measurement_space": MEASUREMENT_SPACES.get(checkpoint, "unknown"),
                            "statistic": statistic,
                        },
                    )

    rows: list[dict[str, Any]] = []
    for key, values in buckets.items():
        available = [value for value in values if value is not None and math.isfinite(value)]
        row = dict(identity[key])
        row.update(
            {
                "n_replicates": len(values),
                "n_available": len(available),
                "n_unavailable": len(values) - len(available),
                "mean": _mean(available),
                "sd": _sd(available),
                "median": _median(available),
            }
        )
        rows.append(row)
    frame = pd.DataFrame(
        rows,
        columns=[
            "trajectory_mode",
            "effect_size",
            "cell_id",
            "phase",
            "checkpoint",
            "scope",
            "measurement_space",
            "statistic",
            "n_replicates",
            "n_available",
            "n_unavailable",
            "mean",
            "sd",
            "median",
        ],
    )
    if frame.empty:
        return frame
    frame["checkpoint_order"] = frame["checkpoint"].map(
        {name: index for index, name in enumerate(CHECKPOINT_ORDER)}
    )
    return frame.sort_values(
        ["trajectory_mode", "effect_size", "checkpoint_order", "scope", "statistic"]
    ).drop(columns=["checkpoint_order"]).reset_index(drop=True)


def _geometry_statistics(values: Mapping[str, Any]) -> Iterable[tuple[str, float | None]]:
    availability = values.get("availability") or {}
    for statistic in _STATISTICS:
        value = values.get(statistic)
        if availability.get(statistic) is False:
            value = None
        yield statistic, None if value is None else float(value)
    for group, length in (values.get("path_lengths") or {}).items():
        yield f"path_length[{group}]", None if length is None else float(length)


# --------------------------------------------------------------------------- #
# 5.2 PLS selection
# --------------------------------------------------------------------------- #


def summarize_pls_selection(records: Sequence[SimulationReplicateResult]) -> pd.DataFrame:
    """Selected component counts, effective CV settings, and AUROC per cell."""

    buckets: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.status != "completed":
            continue
        meta = dict(record.cell_metadata)
        entry = buckets.setdefault(
            record.cell_id,
            {
                "cell_id": record.cell_id,
                "phase": record.phase,
                "trajectory_mode": str(meta.get("trajectory_mode") or "none"),
                "effect_size": meta.get("effect_size"),
                "completed_replicates": 0,
                "missing_integration_metadata": 0,
                "selected_lv": [],
                "cv_mean_auroc": [],
                "cv_settings": set(),
            },
        )
        entry["completed_replicates"] += 1
        integration = record.integration_metadata or {}
        if not integration or integration.get("integration_method") != "pls":
            entry["missing_integration_metadata"] += 1
            continue
        selected = integration.get("selected_lv")
        if selected is None:
            selected = (integration.get("integration_params") or {}).get("selected_lv")
        if selected is None:
            entry["missing_integration_metadata"] += 1
        else:
            entry["selected_lv"].append(int(selected))
        auroc = integration.get("cv_mean_auroc")
        if auroc is not None and math.isfinite(float(auroc)):
            entry["cv_mean_auroc"].append(float(auroc))
        params = integration.get("integration_params") or {}
        entry["cv_settings"].add(
            tuple(
                (key, params.get(key))
                for key in ("cv1_splits", "cv2_splits", "n_repeats", "max_components", "random_state")
            )
        )

    rows: list[dict[str, Any]] = []
    for entry in buckets.values():
        selected = entry.pop("selected_lv")
        auroc = entry.pop("cv_mean_auroc")
        settings = entry.pop("cv_settings")
        row = dict(entry)
        row.update(
            {
                "n_selected_lv": len(selected),
                "selected_lv_mean": _mean(selected),
                "selected_lv_median": _median(selected),
                "selected_lv_mode": _mode(selected),
                "selected_lv_min": min(selected) if selected else None,
                "selected_lv_max": max(selected) if selected else None,
                "cv_mean_auroc_mean": _mean(auroc),
                "cv_settings": _format_settings(settings),
                "cv_settings_consistent": len(settings) <= 1,
            }
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["trajectory_mode", "effect_size", "cell_id"]).reset_index(drop=True)


def _format_settings(settings: set[tuple]) -> str | None:
    if not settings:
        return None
    if len(settings) == 1:
        return "; ".join(f"{key}={value}" for key, value in sorted(next(iter(settings))))
    return f"inconsistent across {len(settings)} settings"


# --------------------------------------------------------------------------- #
# 5.3 attribution stability and truth recovery
# --------------------------------------------------------------------------- #


def summarize_attribution(records: Sequence[SimulationReplicateResult]) -> pd.DataFrame:
    """Attribution availability, retention, stability, and truth recovery.

    Cross-replicate agreement is computed from the persisted top-k identifiers:
    ``top_k_jaccard`` is the mean pairwise overlap of the selected sets across
    replicates, and ``sign_agreement`` is the mean, over features selected in at
    least two replicates, of the fraction agreeing with that feature's modal
    sign. Both are ``None`` when fewer than two replicates carry the transition.
    """

    groups: dict[tuple, dict[str, Any]] = {}
    for record in records:
        if record.status != "completed":
            continue
        meta = dict(record.cell_metadata)
        mode = str(meta.get("trajectory_mode") or "none")
        effect_size = meta.get("effect_size")
        effect_size = float(effect_size) if effect_size is not None else None
        diagnostics = record.attribution_diagnostics or {}
        status = record.attribution_status or diagnostics.get("status") or "not_requested"
        if status == "not_requested":
            continue

        per_transition = {
            (entry["transition_id"], component)
            for entry in diagnostics.get("transitions", [])
            for component in _COMPONENTS
        }
        if not per_transition:
            key: tuple[Any, ...] = (mode, effect_size, record.cell_id, None, None)
            entry_group = _attribution_group(groups, key, mode, effect_size, record)
            entry_group["eligible"] += 1
            entry_group["failed"] += 1 if status == "failed" else 0
            continue

        retention = {entry["transition_id"]: entry.get("retention", {}) for entry in diagnostics.get("transitions", [])}
        recovery = {
            (entry["transition_id"], entry["component"]): entry
            for entry in diagnostics.get("truth_recovery", [])
        }
        stability = {
            (entry["transition_id"], entry["component"]): entry for entry in diagnostics.get("stability", [])
        }
        selected: dict[tuple[str, str], list[tuple[str, int]]] = {}
        for entry in diagnostics.get("top_features", []):
            selected.setdefault((entry["transition_id"], entry["component"]), []).append(
                (str(entry["feature"]), int(entry.get("sign", 0)))
            )

        for transition_id, component in sorted(per_transition):
            key = (mode, effect_size, record.cell_id, transition_id, component)
            entry_group = _attribution_group(groups, key, mode, effect_size, record)
            entry_group["eligible"] += 1
            entry_group["computed"] += 1
            transition_retention = retention.get(transition_id, {})
            if component == "observed" and transition_retention.get("available"):
                _append(entry_group["retention_cosine"], transition_retention.get("cosine"))
                _append(entry_group["retention_norm_ratio"], transition_retention.get("norm_ratio"))
                _append(entry_group["residual_fraction"], transition_retention.get("residual_fraction"))
            recovery_entry = recovery.get((transition_id, component))
            if recovery_entry and recovery_entry.get("available"):
                _append(entry_group["precision"], recovery_entry.get("precision"))
                _append(entry_group["recall"], recovery_entry.get("recall"))
                _append(entry_group["selected_count"], recovery_entry.get("selected_count"))
            stability_entry = stability.get((transition_id, component))
            if stability_entry:
                _append(entry_group["sign_stability"], stability_entry.get("mean_sign_stability"))
                _append(
                    entry_group["top_k_selection_frequency"],
                    stability_entry.get("mean_top_k_selection_frequency"),
                )
            features = selected.get((transition_id, component))
            if features:
                entry_group["selections"].append(features)

    rows: list[dict[str, Any]] = []
    for entry in groups.values():
        selections = entry.pop("selections")
        row = {
            "trajectory_mode": entry["trajectory_mode"],
            "effect_size": entry["effect_size"],
            "cell_id": entry["cell_id"],
            "transition_id": entry["transition_id"],
            "component": entry["component"],
            "eligible_replicates": entry["eligible"],
            "computed_replicates": entry["computed"],
            "failed_replicates": entry["failed"],
            "availability_rate": (entry["computed"] / entry["eligible"]) if entry["eligible"] else None,
            "retention_cosine_mean": _mean(entry["retention_cosine"]),
            "retention_norm_ratio_mean": _mean(entry["retention_norm_ratio"]),
            "residual_fraction_mean": _mean(entry["residual_fraction"]),
            "precision_mean": _mean(entry["precision"]),
            "recall_mean": _mean(entry["recall"]),
            "selected_count_mean": _mean(entry["selected_count"]),
            "bootstrap_sign_stability_mean": _mean(entry["sign_stability"]),
            "bootstrap_top_k_frequency_mean": _mean(entry["top_k_selection_frequency"]),
            "top_k_jaccard": _mean_pairwise_jaccard(selections),
            "sign_agreement": _sign_agreement(selections),
        }
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(
        ["trajectory_mode", "effect_size", "cell_id", "transition_id", "component"],
        na_position="first",
    ).reset_index(drop=True)


def _attribution_group(groups: dict, key: tuple, mode: str, effect_size, record) -> dict[str, Any]:
    return groups.setdefault(
        key,
        {
            "trajectory_mode": mode,
            "effect_size": effect_size,
            "cell_id": record.cell_id,
            "transition_id": key[3],
            "component": key[4],
            "eligible": 0,
            "computed": 0,
            "failed": 0,
            "retention_cosine": [],
            "retention_norm_ratio": [],
            "residual_fraction": [],
            "precision": [],
            "recall": [],
            "selected_count": [],
            "sign_stability": [],
            "top_k_selection_frequency": [],
            "selections": [],
        },
    )


def _mean_pairwise_jaccard(selections: Sequence[Sequence[tuple[str, int]]]) -> float | None:
    sets = [{feature for feature, _ in entry} for entry in selections if entry]
    if len(sets) < 2:
        return None
    scores: list[float] = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            if not union:
                continue
            scores.append(len(sets[i] & sets[j]) / len(union))
    return _mean(scores)


def _sign_agreement(selections: Sequence[Sequence[tuple[str, int]]]) -> float | None:
    if len(selections) < 2:
        return None
    signs: dict[str, list[int]] = {}
    for entry in selections:
        for feature, sign in entry:
            signs.setdefault(feature, []).append(sign)
    scores = [
        Counter(values).most_common(1)[0][1] / len(values)
        for values in signs.values()
        if len(values) >= 2
    ]
    return _mean(scores) if scores else None


# --------------------------------------------------------------------------- #
# 5.4 localization
# --------------------------------------------------------------------------- #


def localize_off_diagonal(
    records: Sequence[SimulationReplicateResult],
    *,
    scope: str = "joint",
    materiality_threshold: float = DEFAULT_MATERIALITY_THRESHOLD,
) -> pd.DataFrame:
    """First checkpoint at which each off-diagonal response becomes material.

    Every checkpoint is judged on its **own** scale-free quantity, so a
    standardized-feature distance is never weighed against a PLS-latent one:

    - ``delta`` is divided by that checkpoint's mean group path length, giving a
      dimensionless relative size difference;
    - ``angle`` is divided by 180, the estimator reporting orientation in
      **degrees** on ``[0, 180]``, giving the fraction of the maximum possible
      orientation difference; and
    - ``shape`` is already a Procrustes distance on centroid-scaled configurations.

    A response is material at a checkpoint when its normalized value exceeds the
    shared zero-effect anchor's normalized value *at that same checkpoint* by at
    least ``materiality_threshold``. Comparing an excess rather than a ratio
    matters: an exactly-null construction leaves the anchor's angle at roughly
    1e-8, and any ratio against that is meaningless.

    The emitted classification labels where a response first appears —
    construction, sampling/preprocessing, or projection. It is descriptive, not
    causal, and gates nothing.
    """

    geometry = summarize_realized_geometry(records)
    columns = [
        "trajectory_mode",
        "effect_size",
        "statistic",
        "first_material_checkpoint",
        "classification",
        "materiality_threshold",
        "normalized_value",
        "normalized_null",
        "normalized_excess",
        "measurement_space",
    ]
    if geometry.empty:
        return pd.DataFrame(columns=columns)

    anchor_cells = {
        record.cell_id
        for record in records
        if record.cell_metadata.get("zero_effect_anchor")
        or (record.phase.startswith("type_i_") and not record.cell_metadata.get("trajectory_mode"))
    }
    scoped = geometry[geometry["scope"] == scope]
    normalized = _normalized_geometry(scoped)
    null_reference = {
        (checkpoint, statistic): value
        for (cell_id, checkpoint, statistic), value in normalized.items()
        if cell_id in anchor_cells
    }

    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, Any, str], dict[str, float]] = {}
    identity: dict[tuple[str, Any, str], dict[str, Any]] = {}
    for row in scoped[scoped["statistic"].isin(_STATISTICS)].itertuples(index=False):
        if row.cell_id in anchor_cells:
            continue
        key = (str(row.trajectory_mode), row.effect_size, str(row.statistic))
        value = normalized.get((str(row.cell_id), str(row.checkpoint), str(row.statistic)))
        if value is not None:
            grouped.setdefault(key, {})[str(row.checkpoint)] = value
        identity.setdefault(key, {"cell_id": str(row.cell_id)})

    for key in sorted(grouped, key=lambda k: (k[0], k[2], k[1] if k[1] is not None else -1.0)):
        mode, effect_size, statistic = key
        if DIAGONAL_STATISTIC.get(mode) == statistic:
            continue
        by_checkpoint = grouped[key]
        first_checkpoint: str | None = None
        observed: float | None = None
        null_value: float | None = None
        excess: float | None = None
        for checkpoint in CHECKPOINT_ORDER:
            value = by_checkpoint.get(checkpoint)
            reference = null_reference.get((checkpoint, statistic))
            if value is None:
                continue
            baseline = 0.0 if reference is None else reference
            candidate_excess = value - baseline
            if candidate_excess >= materiality_threshold:
                first_checkpoint = checkpoint
                observed = value
                null_value = reference
                excess = candidate_excess
                break
        rows.append(
            {
                "trajectory_mode": mode,
                "effect_size": effect_size,
                "statistic": statistic,
                "first_material_checkpoint": first_checkpoint,
                "classification": (
                    CHECKPOINT_CLASSIFICATION.get(first_checkpoint, "unclassified")
                    if first_checkpoint is not None
                    else "not_material"
                ),
                "materiality_threshold": materiality_threshold,
                "normalized_value": observed,
                "normalized_null": null_value,
                "normalized_excess": excess,
                "measurement_space": (
                    MEASUREMENT_SPACES.get(first_checkpoint) if first_checkpoint is not None else None
                ),
            }
        )
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        return frame
    return frame.sort_values(["trajectory_mode", "statistic", "effect_size"]).reset_index(drop=True)


def _normalized_geometry(scoped: pd.DataFrame) -> dict[tuple[str, str, str], float]:
    """Scale-free statistic per (cell, checkpoint, statistic).

    ``delta`` is expressed relative to the mean group path length measured at the
    same checkpoint and ``angle`` relative to its 180-degree maximum; ``shape``
    is already dimensionless.
    """

    path_scale: dict[tuple[str, str], list[float]] = {}
    for row in scoped.itertuples(index=False):
        if not str(row.statistic).startswith("path_length["):
            continue
        value = _as_float(row.mean)
        if value is not None:
            path_scale.setdefault((str(row.cell_id), str(row.checkpoint)), []).append(value)

    out: dict[tuple[str, str, str], float] = {}
    for row in scoped.itertuples(index=False):
        statistic = str(row.statistic)
        if statistic not in _STATISTICS:
            continue
        value = _as_float(row.mean)
        if value is None:
            continue
        key = (str(row.cell_id), str(row.checkpoint), statistic)
        if statistic == "angle":
            out[key] = value / _MAX_ANGLE_DEGREES
            continue
        if statistic != "delta":
            out[key] = value
            continue
        lengths = path_scale.get((str(row.cell_id), str(row.checkpoint)), [])
        scale = _mean(lengths)
        if scale is None or scale <= 0.0:
            continue
        out[key] = value / scale
    return out


# --------------------------------------------------------------------------- #
# 5.5 gate observations
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GateObservation:
    """One explicit gate observation with its Monte Carlo uncertainty."""

    rule: str
    kind: str
    trajectory_mode: str
    statistic: str
    cell_id: str | None
    effect_size: float | None
    met: bool | None
    detail: str
    observations: dict[str, Any] = field(default_factory=dict)


def type_i_inflation_bound(alpha: float, se_tolerance: float, n_available: int) -> float:
    """One-sided inflation bound ``alpha + k * sqrt(alpha (1 - alpha) / n)``."""

    if n_available <= 0:
        raise Phase4SummaryError("n_available must be positive to compute an inflation bound.")
    return float(alpha + se_tolerance * math.sqrt(alpha * (1.0 - alpha) / n_available))


def evaluate_type_i_inflation(
    gate: Phase4GateConfig,
    summaries: Sequence[SimulationSummaryResult],
    records: Sequence[SimulationReplicateResult],
) -> list[GateObservation]:
    """Check every control cell's statistics against the one-sided bound.

    Controls are the ``none`` baseline and every declared control-mode cell at
    every enumerated effect level — translation is a location-only offset at any
    effect, so each of its power-grid levels is a Type I control too.
    """

    cell_meta = _cell_metadata_index(records)
    out: list[GateObservation] = []
    for summary in summaries:
        meta = cell_meta.get(summary.cell_id)
        if meta is None or meta.get("varied_axis") is not None:
            continue
        mode = _resolve_mode(meta)
        if mode not in gate.control_modes:
            continue
        effect_size = meta.get("effect_size")
        base: dict[str, Any] = {
            "cell_id": summary.cell_id,
            "phase": summary.phase,
            "trajectory_mode": mode,
            "statistic": summary.statistic,
            "effect_size": None if effect_size is None else float(effect_size),
            "alpha": gate.alpha,
            "se_tolerance": gate.control_se_tolerance,
            "rejection_rate": summary.rejection_rate,
            "monte_carlo_se": summary.monte_carlo_se,
            "available_replicates": summary.available_replicates,
        }
        name = f"type_i_inflation[{summary.cell_id},{summary.statistic}]"
        if summary.rejection_rate is None or summary.available_replicates <= 0:
            out.append(
                GateObservation(
                    rule=name,
                    kind="type_i_inflation",
                    trajectory_mode=mode,
                    statistic=summary.statistic,
                    cell_id=summary.cell_id,
                    effect_size=base["effect_size"],
                    met=None,
                    detail="No available replicates",
                    observations=base,
                )
            )
            continue
        bound = type_i_inflation_bound(gate.alpha, gate.control_se_tolerance, summary.available_replicates)
        exceedance = float(summary.rejection_rate) - bound
        se = summary.monte_carlo_se or 0.0
        base.update({"bound": bound, "exceedance": exceedance, "exceedance_in_se": _ratio(exceedance, se)})
        met = exceedance <= 0.0
        out.append(
            GateObservation(
                rule=name,
                kind="type_i_inflation",
                trajectory_mode=mode,
                statistic=summary.statistic,
                cell_id=summary.cell_id,
                effect_size=base["effect_size"],
                met=met,
                detail=(
                    f"rate={summary.rejection_rate:.4f} {'<=' if met else '>'} bound={bound:.4f} "
                    f"(alpha={gate.alpha} + {gate.control_se_tolerance:g}·SE, n={summary.available_replicates})"
                ),
                observations=base,
            )
        )
    return out


def evaluate_power_rule(
    gate: Phase4GateConfig,
    rule: GateRule,
    operating: pd.DataFrame,
) -> GateObservation:
    """Power floor at the top effect plus uncertainty-tolerant monotonicity.

    A downward step between adjacent effect levels is tolerated when it is no
    larger than ``monotonicity_se_tolerance`` times the two points' combined
    Monte Carlo standard error, so ordinary sampling noise is not read as a
    broken power curve.
    """

    floor = gate.power_floor(rule)
    observations: dict[str, Any] = {
        "trajectory_mode": rule.trajectory_mode,
        "statistic": rule.statistic,
        "min_power_at_top": floor,
        "monotonicity_se_tolerance": gate.monotonicity_se_tolerance,
    }
    if operating.empty:
        return GateObservation(
            rule=rule.name,
            kind="power",
            trajectory_mode=rule.trajectory_mode,
            statistic=rule.statistic,
            cell_id=None,
            effect_size=None,
            met=None,
            detail="No operating summaries available",
            observations=observations,
        )
    block = operating[
        (operating["trajectory_mode"] == rule.trajectory_mode)
        & (operating["statistic"] == rule.statistic)
        & operating["rejection_rate"].notna()
    ].sort_values("effect_size")
    points: list[dict[str, Any]] = [
        {
            "effect_size": _as_float(row.effect_size),
            "rate": _as_float(row.rejection_rate),
            "se": _as_float(row.monte_carlo_se),
            "from_shared_anchor": bool(row.from_shared_anchor),
        }
        for row in block.itertuples(index=False)
    ]
    points = [point for point in points if point["rate"] is not None and point["effect_size"] is not None]
    observations["points"] = points
    if not points:
        return GateObservation(
            rule=rule.name,
            kind="power",
            trajectory_mode=rule.trajectory_mode,
            statistic=rule.statistic,
            cell_id=None,
            effect_size=None,
            met=None,
            detail="No matching power points",
            observations=observations,
        )

    reversals: list[dict[str, Any]] = []
    for first, second in zip(points, points[1:]):
        drop = first["rate"] - second["rate"]
        if drop <= 0:
            continue
        combined_se = math.sqrt((first["se"] or 0.0) ** 2 + (second["se"] or 0.0) ** 2)
        tolerance = gate.monotonicity_se_tolerance * combined_se
        if drop > tolerance:
            reversals.append(
                {
                    "from_effect": first["effect_size"],
                    "to_effect": second["effect_size"],
                    "drop": drop,
                    "tolerance": tolerance,
                }
            )
    top = points[-1]
    meets_floor = top["rate"] >= floor
    monotone = not reversals
    observations.update(
        {"top_effect_size": top["effect_size"], "top_rate": top["rate"], "reversals": reversals}
    )
    met = monotone and meets_floor
    return GateObservation(
        rule=rule.name,
        kind="power",
        trajectory_mode=rule.trajectory_mode,
        statistic=rule.statistic,
        cell_id=None,
        effect_size=top["effect_size"],
        met=met,
        detail=(
            f"top_rate={top['rate']:.3f} {'>=' if meets_floor else '<'} floor={floor:.3f}; "
            f"tolerated_monotone={monotone}"
            + ("" if monotone else f"; material reversals={reversals}")
        ),
        observations=observations,
    )


def evaluate_control_rule(
    gate: Phase4GateConfig,
    rule: GateRule,
    operating: pd.DataFrame,
) -> GateObservation:
    """A mandatory off-diagonal statistic checked against the inflation bound.

    Used for pairs such as magnitude's ``angle`` and ``shape``, whose
    construction should not move them; the check uses the largest available
    effect level, where any leakage is most visible.
    """

    observations: dict[str, Any] = {
        "trajectory_mode": rule.trajectory_mode,
        "statistic": rule.statistic,
        "alpha": gate.alpha,
        "se_tolerance": gate.control_se_tolerance,
    }
    block = (
        operating[
            (operating["trajectory_mode"] == rule.trajectory_mode)
            & (operating["statistic"] == rule.statistic)
            & operating["rejection_rate"].notna()
            & (operating["available_replicates"] > 0)
        ].sort_values("effect_size")
        if not operating.empty
        else operating
    )
    if block.empty:
        return GateObservation(
            rule=rule.name,
            kind="control",
            trajectory_mode=rule.trajectory_mode,
            statistic=rule.statistic,
            cell_id=None,
            effect_size=None,
            met=None,
            detail="No matching control summaries",
            observations=observations,
        )
    row = block.iloc[-1]
    n_available = int(row["available_replicates"])
    bound = type_i_inflation_bound(gate.alpha, gate.control_se_tolerance, n_available)
    rate = _as_float(row["rejection_rate"]) or 0.0
    se = _as_float(row["monte_carlo_se"]) or 0.0
    exceedance = rate - bound
    observations.update(
        {
            "cell_id": str(row["cell_id"]),
            "effect_size": _as_float(row["effect_size"]),
            "rejection_rate": rate,
            "monte_carlo_se": se,
            "available_replicates": n_available,
            "bound": bound,
            "exceedance": exceedance,
            "exceedance_in_se": _ratio(exceedance, se),
        }
    )
    met = exceedance <= 0.0
    return GateObservation(
        rule=rule.name,
        kind="control",
        trajectory_mode=rule.trajectory_mode,
        statistic=rule.statistic,
        cell_id=str(row["cell_id"]),
        effect_size=_as_float(row["effect_size"]),
        met=met,
        detail=(
            f"rate={rate:.4f} {'<=' if met else '>'} bound={bound:.4f} "
            f"(alpha={gate.alpha} + {gate.control_se_tolerance:g}·SE, n={n_available})"
        ),
        observations=observations,
    )


def evaluate_descriptive_rule(rule: GateRule, operating: pd.DataFrame) -> GateObservation:
    """A reported-but-non-gating mode/statistic pair."""

    block = (
        operating[
            (operating["trajectory_mode"] == rule.trajectory_mode)
            & (operating["statistic"] == rule.statistic)
        ].sort_values("effect_size")
        if not operating.empty
        else operating
    )
    points = (
        []
        if block.empty
        else [
            {
                "effect_size": _as_float(row.effect_size),
                "rate": _as_float(row.rejection_rate),
                "se": _as_float(row.monte_carlo_se),
            }
            for row in block.itertuples(index=False)
        ]
    )
    return GateObservation(
        rule=rule.name,
        kind="descriptive",
        trajectory_mode=rule.trajectory_mode,
        statistic=rule.statistic,
        cell_id=None,
        effect_size=None,
        met=None,
        detail=(
            "Descriptive only: the constructions are known to be mixed, so this pair is reported "
            "against realized geometry rather than gated."
        ),
        observations={"points": points},
    )


# --------------------------------------------------------------------------- #
# 5.6 completeness and the aggregated gate
# --------------------------------------------------------------------------- #


def evaluate_completeness(
    records: Sequence[SimulationReplicateResult],
    *,
    expected_units: int | None = None,
) -> GateObservation:
    """Every expected work unit resolved, with complete mandatory diagnostics."""

    completed = [r for r in records if r.status == "completed"]
    failed = [r for r in records if r.status == "failed"]
    pls = [r for r in completed if (r.integration_metadata or {}).get("integration_method") == "pls"]
    missing_integration = [
        r for r in completed if not r.integration_metadata or r.integration_metadata.get("selected_lv") is None
    ]
    missing_geometry = [r for r in completed if not (r.realized_geometry or {}).get("checkpoints")]
    # Eligibility is recorded on the persisted status: a cell that was never
    # selected reads "not_requested", so anything else was eligible.
    eligible = [r for r in completed if (r.attribution_status or "not_requested") != "not_requested"]
    incomplete_attribution = [
        r
        for r in eligible
        if r.attribution_status not in {"computed", "failed"}
        or (r.attribution_status == "computed" and not r.attribution_diagnostics.get("transitions"))
        or (r.attribution_status == "failed" and not r.diagnostic_error_message)
    ]
    observations = {
        "records": len(records),
        "completed": len(completed),
        "failed": len(failed),
        "expected_units": expected_units,
        "pls_records": len(pls),
        "missing_integration_metadata": len(missing_integration),
        "missing_realized_geometry": len(missing_geometry),
        "attribution_eligible": len(eligible),
        "attribution_incomplete": len(incomplete_attribution),
    }
    problems: list[str] = []
    if expected_units is not None and len(completed) != expected_units:
        problems.append(f"{len(completed)} of {expected_units} expected work units completed")
    if failed:
        problems.append(f"{len(failed)} failed work unit(s)")
    if missing_integration:
        problems.append(f"{len(missing_integration)} completed record(s) lack selected-component metadata")
    if missing_geometry:
        problems.append(f"{len(missing_geometry)} completed record(s) lack realized-geometry metadata")
    if incomplete_attribution:
        problems.append(
            f"{len(incomplete_attribution)} eligible record(s) lack valid attribution diagnostics or a failure reason"
        )
    met = not problems
    return GateObservation(
        rule="diagnostic_completeness",
        kind="completeness",
        trajectory_mode="*",
        statistic="*",
        cell_id=None,
        effect_size=None,
        met=met,
        detail="All expected records and mandatory diagnostics present" if met else "; ".join(problems),
        observations=observations,
    )


@dataclass(frozen=True)
class Phase4GateDecision:
    """Aggregated Phase 4 gate outcome for the paper-grade study."""

    decision: str
    rationale: str
    observations: tuple[GateObservation, ...]
    confirmation_runs: tuple[dict[str, Any], ...] = ()

    def to_frame(self) -> pd.DataFrame:
        """Structured rows for CSV/JSON output."""

        rows = [
            {
                "rule": observation.rule,
                "kind": observation.kind,
                "trajectory_mode": observation.trajectory_mode,
                "statistic": observation.statistic,
                "cell_id": observation.cell_id,
                "effect_size": observation.effect_size,
                "met": observation.met,
                "detail": observation.detail,
                **{f"obs_{key}": value for key, value in observation.observations.items()},
            }
            for observation in self.observations
        ]
        return pd.DataFrame(rows)


def evaluate_phase4_gate(
    gate: Phase4GateConfig,
    summaries: Sequence[SimulationSummaryResult],
    records: Sequence[SimulationReplicateResult],
    *,
    expected_units: int | None = None,
) -> Phase4GateDecision:
    """Aggregate the predeclared gates into ``proceed``/``hold``/``indeterminate``.

    Gate multiplicity is predeclared, not discovered after the run: each control
    cell contributes one one-sided test per statistic, and at a true rate of
    exactly alpha each exceeds its bound with probability around 0.023. One
    marginal exceedance therefore must not decide the phase. When exactly one
    control statistic exceeds its bound by less than
    ``confirmation_se_threshold`` Monte Carlo standard errors and no other
    mandatory gate fails, the decision is ``indeterminate`` with a named
    confirmation re-run. Two or more exceedances, or any exceedance of at least
    one standard error, is ``hold``.

    ``proceed`` is never emitted when mandatory records or diagnostics are
    incomplete.
    """

    if not gate.enabled:
        raise Phase4SummaryError(
            "Phase 4 gate evaluation requires acceptance.gate.enabled in the study configuration."
        )
    operating = build_operating_frame(summaries, records)
    observations: list[GateObservation] = list(evaluate_type_i_inflation(gate, summaries, records))
    for rule in gate.rules:
        if rule.role == "mandatory_power":
            observations.append(evaluate_power_rule(gate, rule, operating))
        elif rule.role == "mandatory_control":
            observations.append(evaluate_control_rule(gate, rule, operating))
        else:
            observations.append(evaluate_descriptive_rule(rule, operating))
    completeness = evaluate_completeness(records, expected_units=expected_units)
    if gate.require_complete_records:
        observations.append(completeness)

    control_failures = [
        observation
        for observation in observations
        if observation.kind in {"type_i_inflation", "control"} and observation.met is False
    ]
    marginal = [
        observation
        for observation in control_failures
        if _is_marginal(observation, gate.confirmation_se_threshold)
    ]
    material = [observation for observation in control_failures if observation not in marginal]
    other_failures = [
        observation
        for observation in observations
        if observation.kind in {"power", "completeness"} and observation.met is False
    ]
    missing_evidence = [
        observation
        for observation in observations
        if observation.kind in {"power", "type_i_inflation", "control", "completeness"}
        and observation.met is None
    ]

    confirmation_runs: tuple[dict[str, Any], ...] = ()
    if material or other_failures:
        decision = "hold"
        rationale = "Mandatory gate(s) failed: " + "; ".join(
            f"{observation.rule} ({observation.detail})" for observation in material + other_failures
        )
        if len(marginal) > gate.max_marginal_exceedances:
            rationale += (
                f"; plus {len(marginal)} marginal control exceedance(s) beyond the "
                f"{gate.max_marginal_exceedances} allowed"
            )
    elif len(marginal) > gate.max_marginal_exceedances:
        decision = "hold"
        rationale = (
            f"{len(marginal)} control statistics exceed their inflation bounds, above the "
            f"{gate.max_marginal_exceedances} marginal exceedance(s) the confirmation rule allows: "
            + "; ".join(f"{observation.rule} ({observation.detail})" for observation in marginal)
        )
    elif marginal:
        observation = marginal[0]
        decision = "indeterminate"
        confirmation_runs = (
            {
                "cell_id": observation.cell_id,
                "statistic": observation.statistic,
                "trajectory_mode": observation.trajectory_mode,
                "effect_size": observation.effect_size,
                "reason": "single marginal control exceedance awaiting confirmation",
                "action": (
                    f"re-run cell {observation.cell_id} with independent seeds and re-evaluate "
                    f"the {observation.statistic} control before any Phase 5 decision"
                ),
            },
        )
        rationale = (
            f"Exactly one control statistic ({observation.rule}) exceeds its bound by less than "
            f"{gate.confirmation_se_threshold:g} Monte Carlo SE and no other mandatory gate fails; "
            "a confirmation re-run is required before a Phase 5 decision."
        )
    elif missing_evidence:
        decision = "indeterminate"
        rationale = "Incomplete evidence: " + "; ".join(
            f"{observation.rule} ({observation.detail})" for observation in missing_evidence
        )
    else:
        decision = "proceed"
        rationale = "All mandatory gates met with complete eligible diagnostics."

    return Phase4GateDecision(
        decision=decision,
        rationale=rationale,
        observations=tuple(observations),
        confirmation_runs=confirmation_runs,
    )


def _is_marginal(observation: GateObservation, threshold: float) -> bool:
    ratio = observation.observations.get("exceedance_in_se")
    if ratio is None:
        return False
    return float(ratio) < threshold


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _cell_metadata_index(records: Iterable[SimulationReplicateResult]) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for record in records:
        if record.cell_id in index:
            continue
        index[record.cell_id] = dict(record.cell_metadata)
    return index


def _resolve_mode(meta: Mapping[str, Any]) -> str:
    mode = meta.get("trajectory_mode")
    return str(mode) if mode is not None else "none"


def _as_float(value: Any) -> float | None:
    """Coerce a loosely typed tabular value to a finite float, or ``None``."""

    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _append(target: list[float], value: Any) -> None:
    if value is None:
        return
    numeric = float(value)
    if math.isfinite(numeric):
        target.append(numeric)


def _mean(values: Sequence[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return float((ordered[middle - 1] + ordered[middle]) / 2)


def _sd(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return float(math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1)))


def _mode(values: Sequence[int]) -> int | None:
    if not values:
        return None
    counts = Counter(values)
    best = max(counts.values())
    return min(value for value, count in counts.items() if count == best)


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    return float(numerator / denominator)


__all__ = [
    "CHECKPOINT_CLASSIFICATION",
    "CHECKPOINT_ORDER",
    "DEFAULT_MATERIALITY_THRESHOLD",
    "DIAGONAL_STATISTIC",
    "MEASUREMENT_SPACES",
    "GateObservation",
    "Phase4GateDecision",
    "Phase4SummaryError",
    "build_operating_frame",
    "evaluate_completeness",
    "evaluate_control_rule",
    "evaluate_descriptive_rule",
    "evaluate_phase4_gate",
    "evaluate_power_rule",
    "evaluate_type_i_inflation",
    "localize_off_diagonal",
    "summarize_attribution",
    "summarize_pls_selection",
    "summarize_realized_geometry",
    "type_i_inflation_bound",
]
