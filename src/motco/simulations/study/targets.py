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
    DesignPointDecisionRule,
    PowerMonotonicityTarget,
    SpecificityTarget,
    TypeIControlTarget,
)
from motco.simulations.study.spectrum import record_design_point


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


# ── Design-point decision ─────────────────────────────────────────────────────

#: Per-column classifications, in the order they are reported.
DESIGN_POINT_STATUSES: tuple[str, ...] = ("meets", "marginal", "fails", "unavailable")


@dataclass(frozen=True)
class DesignPointColumn:
    """One design point's standing against the predeclared rule."""

    design_point: dict[str, Any]
    is_baseline: bool
    top_effect: float | None
    n_available: int
    rejection_rate: float | None
    monte_carlo_se: float | None
    lower_bound: float | None
    status: str
    anchor_rates: dict[str, float | None]
    anchor_available: int


@dataclass(frozen=True)
class DesignPointDecision:
    """Outcome of the design-point rule over every enumerated design point.

    ``verdict`` is ``"chosen"`` when a confirmed column exists (``chosen`` names
    it), ``"revise_claim"`` when none is confirmed, and ``"no_design_grid"``
    when the records carry no design points at all. Advisory only: it feeds
    neither the Phase 4 gate nor the acceptance-target report.
    """

    rule: dict[str, Any]
    alpha: float
    verdict: str
    chosen: dict[str, Any] | None
    rationale: str
    columns: tuple[DesignPointColumn, ...]

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for column in self.columns:
            row: dict[str, Any] = {**column.design_point}
            row.update(
                {
                    "is_baseline": column.is_baseline,
                    "top_effect": column.top_effect,
                    "n_available": column.n_available,
                    "rejection_rate": column.rejection_rate,
                    "monte_carlo_se": column.monte_carlo_se,
                    "lower_bound": column.lower_bound,
                    "status": column.status,
                    "chosen": self.chosen is not None and column.design_point == self.chosen,
                    "anchor_available": column.anchor_available,
                }
            )
            for statistic, rate in column.anchor_rates.items():
                row[f"anchor_{statistic}_rate"] = rate
            rows.append(row)
        return pd.DataFrame(rows)


def evaluate_design_point_decision(
    records: Sequence[SimulationReplicateResult],
    rule: DesignPointDecisionRule,
    *,
    alpha: float = 0.05,
    statistics: Sequence[str] = ("delta", "angle", "shape"),
) -> DesignPointDecision:
    """Classify every design point against ``rule`` and pick the first confirmed one.

    For each design point the target statistic's rejection rate is taken at the
    **largest effect enumerated in that column** for the target mode. A column
    *meets* the floor when ``rate − k·SE ≥ floor`` (``k`` the rule's
    confirmation threshold), is *marginal* when ``rate ≥ floor`` without
    confirmation, *fails* below the floor, and is *unavailable* when no target
    record exists. Columns are ordered by the rule's ``prefer`` axes (ascending,
    in the declared order) and then by the remaining design axes; the chosen
    point is the first *meets* column in that order. Every threshold comes from
    ``rule``; nothing is hard-coded here.
    """

    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1.")

    by_point: dict[tuple[tuple[str, Any], ...], list[SimulationReplicateResult]] = {}
    axes: list[str] = []
    for record in records:
        if record.status != "completed":
            continue
        point = record_design_point(record)
        if point is None:
            continue
        for axis in point:
            if axis not in axes:
                axes.append(axis)
        by_point.setdefault(tuple(point.items()), []).append(record)

    rule_payload = {
        "trajectory_mode": rule.trajectory_mode,
        "statistic": rule.statistic,
        "min_power_at_top": rule.min_power_at_top,
        "confirmation_se_threshold": rule.confirmation_se_threshold,
        "prefer": list(rule.prefer),
    }
    if not by_point or not any(r.phase == "power_design" for group in by_point.values() for r in group):
        return DesignPointDecision(
            rule=rule_payload,
            alpha=alpha,
            verdict="no_design_grid",
            chosen=None,
            rationale="No design-point records are present; the rule has nothing to evaluate.",
            columns=(),
        )

    order_axes = [axis for axis in rule.prefer if axis in axes] + [
        axis for axis in axes if axis not in rule.prefer
    ]
    columns: list[DesignPointColumn] = []
    for coords, group in by_point.items():
        point = dict(coords)
        target = [
            r
            for r in group
            if str(r.cell_metadata.get("trajectory_mode")) == rule.trajectory_mode
            and r.cell_metadata.get("effect_size") is not None
            and float(r.cell_metadata["effect_size"]) != 0.0
        ]
        anchors = [
            r
            for r in group
            if r.cell_metadata.get("effect_size") is not None
            and float(r.cell_metadata["effect_size"]) == 0.0
        ]
        anchor_rates: dict[str, float | None] = {}
        anchor_available = 0
        for statistic in statistics:
            p_values = _finite_p_values(anchors, statistic)
            anchor_rates[statistic] = (
                sum(p < alpha for p in p_values) / len(p_values) if p_values else None
            )
            anchor_available = max(anchor_available, len(p_values))

        if not target:
            columns.append(
                DesignPointColumn(
                    design_point=point,
                    is_baseline=all(r.phase == "power_primary" for r in group),
                    top_effect=None,
                    n_available=0,
                    rejection_rate=None,
                    monte_carlo_se=None,
                    lower_bound=None,
                    status="unavailable",
                    anchor_rates=anchor_rates,
                    anchor_available=anchor_available,
                )
            )
            continue

        top_effect = max(float(r.cell_metadata["effect_size"]) for r in target)
        at_top = [r for r in target if float(r.cell_metadata["effect_size"]) == top_effect]
        p_values = _finite_p_values(at_top, rule.statistic)
        if not p_values:
            rate = se = lower = None
            status = "unavailable"
        else:
            rate = sum(p < alpha for p in p_values) / len(p_values)
            se = math.sqrt(rate * (1.0 - rate) / len(p_values))
            lower = rate - rule.confirmation_se_threshold * se
            if lower >= rule.min_power_at_top:
                status = "meets"
            elif rate >= rule.min_power_at_top:
                status = "marginal"
            else:
                status = "fails"
        columns.append(
            DesignPointColumn(
                design_point=point,
                is_baseline=all(r.phase == "power_primary" for r in group),
                top_effect=top_effect,
                n_available=len(p_values),
                rejection_rate=rate,
                monte_carlo_se=se,
                lower_bound=lower,
                status=status,
                anchor_rates=anchor_rates,
                anchor_available=anchor_available,
            )
        )

    columns.sort(key=lambda column: tuple(_orderable(column.design_point.get(axis)) for axis in order_axes))
    chosen = next((column for column in columns if column.status == "meets"), None)
    if chosen is not None:
        verdict = "chosen"
        rationale = (
            f"First design point in preference order {order_axes} whose {rule.trajectory_mode}/"
            f"{rule.statistic} power at its top effect ({chosen.top_effect:g}) clears the "
            f"{rule.min_power_at_top:.2f} floor with {rule.confirmation_se_threshold:g}·SE "
            f"confirmation: rate {chosen.rejection_rate:.3f}, lower bound {chosen.lower_bound:.3f}."
        )
    else:
        verdict = "revise_claim"
        counts = {status: sum(c.status == status for c in columns) for status in DESIGN_POINT_STATUSES}
        rationale = (
            f"No design point clears the {rule.min_power_at_top:.2f} floor for {rule.trajectory_mode}/"
            f"{rule.statistic} with {rule.confirmation_se_threshold:g}·SE confirmation "
            f"({counts['marginal']} marginal, {counts['fails']} failing, {counts['unavailable']} "
            "unavailable). Per the readiness worklist, revise the method or the claim — not the "
            "Monte Carlo size."
        )
    return DesignPointDecision(
        rule=rule_payload,
        alpha=alpha,
        verdict=verdict,
        chosen=None if chosen is None else dict(chosen.design_point),
        rationale=rationale,
        columns=tuple(columns),
    )


def write_design_point_decision(decision: DesignPointDecision, out_dir: Path) -> dict[str, Path]:
    """Write the design-point decision as JSON (verdict + columns) and CSV (columns)."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "design_point_decision.json"
    csv_path = out_dir / "design_point_decision.csv"
    payload = {
        "verdict": decision.verdict,
        "chosen": decision.chosen,
        "rationale": decision.rationale,
        "alpha": decision.alpha,
        "rule": decision.rule,
        "columns": [asdict(column) for column in decision.columns],
    }
    json_path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")
    decision.to_frame().to_csv(csv_path, index=False)
    return {"design_point_decision": json_path, "design_point_decision_csv": csv_path}


def _finite_p_values(records: Sequence[SimulationReplicateResult], statistic: str) -> list[float]:
    return [
        float(p)
        for record in records
        if (p := record.p_values.get(statistic)) is not None and math.isfinite(float(p))
    ]


def _orderable(value: Any) -> tuple[int, Any]:
    if isinstance(value, bool | int | float):
        return (0, float(value))
    return (1, str(value))


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
    "DESIGN_POINT_STATUSES",
    "DesignPointColumn",
    "DesignPointDecision",
    "TargetEvaluation",
    "evaluate_design_point_decision",
    "evaluate_targets",
    "write_design_point_decision",
    "write_target_report",
]
