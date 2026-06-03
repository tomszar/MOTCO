from __future__ import annotations

import json
from pathlib import Path

from motco.simulations import SimulationReplicateResult, SimulationSummaryResult
from motco.simulations.study.config import (
    AcceptanceTargets,
    PowerMonotonicityTarget,
    SpecificityTarget,
    TypeIControlTarget,
)
from motco.simulations.study.targets import evaluate_targets, write_target_report


def _summary(
    cell_id: str,
    phase: str,
    statistic: str,
    rate: float,
    se: float,
    available: int = 100,
) -> SimulationSummaryResult:
    return SimulationSummaryResult(
        cell_id=cell_id,
        phase=phase,
        statistic=statistic,
        alpha=0.05,
        completed_replicates=available,
        available_replicates=available,
        rejected_replicates=int(rate * available),
        rejection_rate=rate,
        monte_carlo_se=se,
        unavailable_replicates=0,
    )


def _record(
    cell_id: str,
    phase: str,
    mode: str | None,
    effect_size: float | None,
    varied_axis=None,
) -> SimulationReplicateResult:
    metadata = {"varied_axis": varied_axis}
    if mode is not None:
        metadata["trajectory_mode"] = mode
    if effect_size is not None:
        metadata["effect_size"] = effect_size
    return SimulationReplicateResult(
        cell_id=cell_id,
        phase=phase,
        replicate_index=0,
        replicate_seed=0,
        generator_seed=0,
        evaluation_seed=0,
        parameter_signature="sig",
        status="completed",
        cell_metadata=metadata,
    )


def test_type_i_control_target_met_and_not_met() -> None:
    summaries = [
        _summary("null-met", "type_i_baseline", "delta", 0.06, 0.02),
        _summary("null-fail", "type_i_baseline", "delta", 0.20, 0.02),
    ]
    records = [
        _record("null-met", "type_i_baseline", mode=None, effect_size=None),
        _record("null-fail", "type_i_baseline", mode=None, effect_size=None),
    ]
    acceptance = AcceptanceTargets(type_i=(TypeIControlTarget(alpha=0.05, se_tolerance=2.0),))
    evaluations = evaluate_targets(acceptance, summaries, records)
    by_cell = {ev.observations["cell_id"]: ev for ev in evaluations}
    assert by_cell["null-met"].met is True
    assert by_cell["null-fail"].met is False


def test_power_monotonicity_target_met() -> None:
    summaries = [
        _summary("magn-0.1", "power_primary", "delta", 0.20, 0.04),
        _summary("magn-0.5", "power_primary", "delta", 0.85, 0.03),
    ]
    records = [
        _record("magn-0.1", "power_primary", mode="magnitude", effect_size=0.1),
        _record("magn-0.5", "power_primary", mode="magnitude", effect_size=0.5),
    ]
    acceptance = AcceptanceTargets(
        power=(PowerMonotonicityTarget(trajectory_mode="magnitude", statistic="delta", min_power_at_top=0.80),)
    )
    [evaluation] = evaluate_targets(acceptance, summaries, records)
    assert evaluation.met is True
    assert evaluation.observations["monotone"] is True
    assert evaluation.observations["top_rate"] == 0.85


def test_power_monotonicity_target_not_met_due_to_floor() -> None:
    summaries = [
        _summary("magn-0.1", "power_primary", "delta", 0.20, 0.04),
        _summary("magn-0.5", "power_primary", "delta", 0.55, 0.03),
    ]
    records = [
        _record("magn-0.1", "power_primary", mode="magnitude", effect_size=0.1),
        _record("magn-0.5", "power_primary", mode="magnitude", effect_size=0.5),
    ]
    acceptance = AcceptanceTargets(
        power=(PowerMonotonicityTarget(trajectory_mode="magnitude", statistic="delta", min_power_at_top=0.80),)
    )
    [evaluation] = evaluate_targets(acceptance, summaries, records)
    assert evaluation.met is False


def test_power_monotonicity_target_not_met_due_to_non_monotone() -> None:
    summaries = [
        _summary("magn-0.1", "power_primary", "delta", 0.95, 0.02),
        _summary("magn-0.5", "power_primary", "delta", 0.40, 0.03),
    ]
    records = [
        _record("magn-0.1", "power_primary", mode="magnitude", effect_size=0.1),
        _record("magn-0.5", "power_primary", mode="magnitude", effect_size=0.5),
    ]
    acceptance = AcceptanceTargets(
        power=(PowerMonotonicityTarget(trajectory_mode="magnitude", statistic="delta", min_power_at_top=0.30),)
    )
    [evaluation] = evaluate_targets(acceptance, summaries, records)
    assert evaluation.met is False
    assert evaluation.observations["monotone"] is False


def test_specificity_target_uses_largest_effect_size_and_met() -> None:
    summaries = [
        _summary("translation-0.1", "power_primary", "angle", 0.06, 0.02),
        _summary("translation-0.5", "power_primary", "angle", 0.07, 0.02),
    ]
    records = [
        _record("translation-0.1", "power_primary", mode="translation", effect_size=0.1),
        _record("translation-0.5", "power_primary", mode="translation", effect_size=0.5),
    ]
    acceptance = AcceptanceTargets(
        specificity=(SpecificityTarget(trajectory_mode="translation", statistic="angle", alpha=0.05, se_tolerance=2.0),)
    )
    [evaluation] = evaluate_targets(acceptance, summaries, records)
    assert evaluation.met is True
    assert evaluation.observations["effect_size"] == 0.5


def test_specificity_target_not_met_when_far_from_alpha() -> None:
    summaries = [_summary("translation-0.5", "power_primary", "angle", 0.50, 0.02)]
    records = [_record("translation-0.5", "power_primary", mode="translation", effect_size=0.5)]
    acceptance = AcceptanceTargets(
        specificity=(SpecificityTarget(trajectory_mode="translation", statistic="angle", alpha=0.05, se_tolerance=2.0),)
    )
    [evaluation] = evaluate_targets(acceptance, summaries, records)
    assert evaluation.met is False


def test_write_target_report_emits_csv_and_json(tmp_path: Path) -> None:
    summaries = [_summary("null-met", "type_i_baseline", "delta", 0.05, 0.02)]
    records = [_record("null-met", "type_i_baseline", mode=None, effect_size=None)]
    acceptance = AcceptanceTargets(type_i=(TypeIControlTarget(alpha=0.05),))
    evaluations = evaluate_targets(acceptance, summaries, records)
    paths = write_target_report(evaluations, tmp_path)
    assert paths["csv"].exists()
    assert paths["json"].exists()
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload[0]["target_kind"] == "type_i_control"
