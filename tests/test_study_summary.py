from __future__ import annotations

import pytest

from motco.simulations import SimulationReplicateResult
from motco.simulations.study.summary import (
    summarize_combined_rule,
    summarize_study,
)


def _record(
    cell_id: str,
    phase: str,
    replicate_index: int,
    p_values: dict,
    trajectory_mode: str | None = None,
) -> SimulationReplicateResult:
    return SimulationReplicateResult(
        cell_id=cell_id,
        phase=phase,
        replicate_index=replicate_index,
        replicate_seed=replicate_index,
        generator_seed=replicate_index,
        evaluation_seed=replicate_index,
        parameter_signature="sig",
        status="completed",
        p_values=p_values,
        cell_metadata={"trajectory_mode": trajectory_mode} if trajectory_mode else {},
    )


def test_summarize_study_handles_unavailable_statistic() -> None:
    records = [
        _record("c", "type_i_baseline", 0, {"delta": 0.01, "angle": 0.3}),
        _record("c", "type_i_baseline", 1, {"delta": 0.10, "angle": 0.02}),
    ]
    summaries = summarize_study(records, alpha=0.05)
    by_stat = {s.statistic: s for s in summaries}
    assert by_stat["delta"].rejection_rate == 0.5
    assert by_stat["angle"].rejection_rate == 0.5
    assert by_stat["shape"].available_replicates == 0
    assert by_stat["shape"].rejection_rate is None
    assert by_stat["shape"].unavailable_replicates == 2


def test_combined_rule_rejects_when_any_significant() -> None:
    records = [
        # rep 0: only delta significant
        _record("c", "type_i_baseline", 0, {"delta": 0.01, "angle": 0.3, "shape": 0.5}),
        # rep 1: only angle significant
        _record("c", "type_i_baseline", 1, {"delta": 0.20, "angle": 0.02, "shape": 0.6}),
        # rep 2: nothing significant
        _record("c", "type_i_baseline", 2, {"delta": 0.40, "angle": 0.20, "shape": 0.6}),
        # rep 3: shape significant (and unavailable in below variant test)
        _record("c", "type_i_baseline", 3, {"delta": 0.40, "angle": 0.20, "shape": 0.01}),
    ]
    combined = summarize_combined_rule(records, alpha=0.05)
    assert len(combined) == 1
    summary = combined[0]
    assert summary.rejected_replicates == 3
    assert summary.available_replicates == 4
    assert summary.rejection_rate == 0.75


def test_combined_rule_skips_power_phase_by_default() -> None:
    records = [
        _record("c", "power_primary", 0, {"delta": 0.01}, trajectory_mode="magnitude"),
        _record("c", "type_i_baseline", 1, {"delta": 0.01}),
    ]
    combined = summarize_combined_rule(records, alpha=0.05)
    assert len(combined) == 1
    assert combined[0].phase == "type_i_baseline"


def test_combined_rule_recognizes_translation_negative_control() -> None:
    records = [
        _record("c", "power_primary", 0, {"delta": 0.01}, trajectory_mode="translation"),
        _record("c", "power_primary", 1, {"delta": 0.20}, trajectory_mode="translation"),
    ]
    combined = summarize_combined_rule(records, alpha=0.05)
    assert len(combined) == 1
    assert combined[0].rejection_rate == 0.5


def test_combined_rule_unavailable_when_all_pvalues_missing() -> None:
    records = [
        _record("c", "type_i_baseline", 0, {"delta": None, "angle": None, "shape": None}),
    ]
    combined = summarize_combined_rule(records, alpha=0.05)
    assert combined[0].available_replicates == 0
    assert combined[0].rejection_rate is None


def test_summarize_study_rejects_invalid_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        summarize_study([], alpha=0.0)
