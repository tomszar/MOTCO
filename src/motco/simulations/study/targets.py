"""Evaluate study summaries against pre-specified acceptance targets."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from motco.simulations.grid import (
    SimulationReplicateResult,
    SimulationSummaryResult,
)
from motco.simulations.study.config import (
    AcceptanceTargets,
    PowerMonotonicityTarget,
    SpecificityTarget,
    TypeIControlTarget,
)


@dataclass(frozen=True)
class TargetEvaluation:
    """Result of evaluating one acceptance target against summaries."""

    target_name: str
    target_kind: str
    met: bool | None
    rationale: str
    observations: dict[str, Any]


def evaluate_targets(
    acceptance: AcceptanceTargets,
    summaries: Sequence[SimulationSummaryResult],
    records: Sequence[SimulationReplicateResult],
) -> list[TargetEvaluation]:
    """Evaluate every acceptance target against the supplied summaries."""

    cell_meta = _cell_metadata_index(records)
    evaluations: list[TargetEvaluation] = []
    for type_i_target in acceptance.type_i:
        evaluations.extend(_evaluate_type_i(type_i_target, summaries, cell_meta))
    for power_target in acceptance.power:
        evaluations.append(_evaluate_power(power_target, summaries, cell_meta))
    for specificity_target in acceptance.specificity:
        evaluations.append(_evaluate_specificity(specificity_target, summaries, cell_meta))
    return evaluations


def write_target_report(
    evaluations: Sequence[TargetEvaluation],
    out_dir: Path,
) -> dict[str, Path]:
    """Write acceptance target evaluations as CSV + JSON."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "acceptance_report.csv"
    json_path = out_dir / "acceptance_report.json"
    rows = []
    for ev in evaluations:
        row = {
            "target_name": ev.target_name,
            "target_kind": ev.target_kind,
            "met": ev.met,
            "rationale": ev.rationale,
        }
        for key, value in ev.observations.items():
            row[f"obs_{key}"] = value
        rows.append(row)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps([asdict(ev) for ev in evaluations], indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return {"csv": csv_path, "json": json_path}


def _evaluate_type_i(
    target: TypeIControlTarget,
    summaries: Sequence[SimulationSummaryResult],
    cell_meta: dict[str, dict],
) -> list[TargetEvaluation]:
    """Type I target: per null cell × statistic, |rate - alpha| ≤ k · SE."""

    out: list[TargetEvaluation] = []
    for summary in summaries:
        if not summary.phase.startswith("type_i_"):
            continue
        meta = cell_meta.get(summary.cell_id, {})
        observations = {
            "cell_id": summary.cell_id,
            "phase": summary.phase,
            "trajectory_mode": meta.get("trajectory_mode"),
            "statistic": summary.statistic,
            "alpha_target": target.alpha,
            "se_tolerance": target.se_tolerance,
            "rejection_rate": summary.rejection_rate,
            "monte_carlo_se": summary.monte_carlo_se,
            "available_replicates": summary.available_replicates,
        }
        if summary.rejection_rate is None or summary.monte_carlo_se is None:
            out.append(
                TargetEvaluation(
                    target_name=f"{target.name}[{summary.cell_id},{summary.statistic}]",
                    target_kind=target.kind,
                    met=None,
                    rationale="No available replicates",
                    observations=observations,
                )
            )
            continue
        bound = target.se_tolerance * summary.monte_carlo_se
        deviation = abs(summary.rejection_rate - target.alpha)
        met = deviation <= bound
        rationale = (
            f"|rate - alpha| = {deviation:.4f} {'<=' if met else '>'} "
            f"{target.se_tolerance:.2f}·SE ({bound:.4f})"
        )
        observations["deviation"] = deviation
        observations["bound"] = bound
        out.append(
            TargetEvaluation(
                target_name=f"{target.name}[{summary.cell_id},{summary.statistic}]",
                target_kind=target.kind,
                met=met,
                rationale=rationale,
                observations=observations,
            )
        )
    return out


def _evaluate_power(
    target: PowerMonotonicityTarget,
    summaries: Sequence[SimulationSummaryResult],
    cell_meta: dict[str, dict],
) -> TargetEvaluation:
    """Power target: rate non-decreasing in effect_size and ≥ floor at top effect."""

    points: list[tuple[float, float, float | None]] = []
    for summary in summaries:
        meta = cell_meta.get(summary.cell_id, {})
        if meta.get("varied_axis") is not None:
            continue
        if meta.get("trajectory_mode") != target.trajectory_mode:
            continue
        if summary.statistic != target.statistic:
            continue
        effect_size = meta.get("effect_size")
        if effect_size is None or summary.rejection_rate is None:
            continue
        points.append((float(effect_size), float(summary.rejection_rate), summary.monte_carlo_se))
    points.sort(key=lambda x: x[0])
    observations: dict[str, Any] = {
        "trajectory_mode": target.trajectory_mode,
        "statistic": target.statistic,
        "min_power_at_top": target.min_power_at_top,
        "points": [{"effect_size": e, "rate": r, "se": s} for e, r, s in points],
    }
    if not points:
        return TargetEvaluation(
            target_name=target.name,
            target_kind=target.kind,
            met=None,
            rationale="No matching power summaries",
            observations=observations,
        )
    rates = [r for _, r, _ in points]
    monotone = all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))
    top_rate = rates[-1]
    meets_floor = top_rate >= target.min_power_at_top
    met = monotone and meets_floor
    rationale = (
        f"monotone={monotone}; top_rate={top_rate:.3f} "
        f"{'>=' if meets_floor else '<'} floor={target.min_power_at_top:.3f}"
    )
    observations["monotone"] = monotone
    observations["top_rate"] = top_rate
    return TargetEvaluation(
        target_name=target.name,
        target_kind=target.kind,
        met=met,
        rationale=rationale,
        observations=observations,
    )


def _evaluate_specificity(
    target: SpecificityTarget,
    summaries: Sequence[SimulationSummaryResult],
    cell_meta: dict[str, dict],
) -> TargetEvaluation:
    """Specificity target: off-diagonal rate ≈ alpha within k · SE.

    Uses the largest effect-size cell for the given (mode, statistic) pair.
    """

    candidates: list[tuple[float, SimulationSummaryResult]] = []
    for summary in summaries:
        meta = cell_meta.get(summary.cell_id, {})
        if meta.get("varied_axis") is not None:
            continue
        if meta.get("trajectory_mode") != target.trajectory_mode:
            continue
        if summary.statistic != target.statistic:
            continue
        effect_size = meta.get("effect_size")
        if effect_size is None:
            continue
        candidates.append((float(effect_size), summary))
    candidates.sort(key=lambda x: x[0])
    observations: dict[str, Any] = {
        "trajectory_mode": target.trajectory_mode,
        "statistic": target.statistic,
        "alpha_target": target.alpha,
        "se_tolerance": target.se_tolerance,
    }
    if not candidates:
        return TargetEvaluation(
            target_name=target.name,
            target_kind=target.kind,
            met=None,
            rationale="No matching specificity summaries",
            observations=observations,
        )
    effect_size, summary = candidates[-1]
    observations.update(
        {
            "effect_size": effect_size,
            "cell_id": summary.cell_id,
            "rejection_rate": summary.rejection_rate,
            "monte_carlo_se": summary.monte_carlo_se,
            "available_replicates": summary.available_replicates,
        }
    )
    if summary.rejection_rate is None or summary.monte_carlo_se is None:
        return TargetEvaluation(
            target_name=target.name,
            target_kind=target.kind,
            met=None,
            rationale="No available replicates at top effect size",
            observations=observations,
        )
    bound = target.se_tolerance * summary.monte_carlo_se
    deviation = abs(summary.rejection_rate - target.alpha)
    met = deviation <= bound
    rationale = (
        f"|rate - alpha| = {deviation:.4f} {'<=' if met else '>'} "
        f"{target.se_tolerance:.2f}·SE ({bound:.4f})"
    )
    observations["deviation"] = deviation
    observations["bound"] = bound
    return TargetEvaluation(
        target_name=target.name,
        target_kind=target.kind,
        met=met,
        rationale=rationale,
        observations=observations,
    )


def _cell_metadata_index(records: Sequence[SimulationReplicateResult]) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for record in records:
        if record.cell_id in index:
            continue
        index[record.cell_id] = dict(record.cell_metadata)
    return index


def _json_default(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


__all__ = [
    "TargetEvaluation",
    "evaluate_targets",
    "write_target_report",
]
