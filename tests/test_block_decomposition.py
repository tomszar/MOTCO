"""Block decomposition of a mode's realized geometry.

Separates a construction that is impure *within* an omic block from one that is
impure only *across* blocks — the distinction that explains the Phase 5 magnitude
control failures.
"""

from __future__ import annotations

import pytest

from motco.simulations.grid import SimulationReplicateResult
from motco.simulations.specificity import (
    decompose_block_response,
    summarize_block_localization,
)


def _scope(delta: float, angle: float | None, shape: float | None) -> dict:
    return {
        "path_lengths": {"A": 3.0, "B": 3.0},
        "delta": delta,
        "angle": angle,
        "shape": shape,
        "availability": {
            "delta": True,
            "angle": angle is not None,
            "shape": shape is not None,
        },
    }


def _record(
    cell_id: str,
    *,
    mode: str | None,
    effect: float | None,
    checkpoints: dict,
    phase: str = "power_primary",
    anchor: bool = False,
    index: int = 0,
) -> SimulationReplicateResult:
    metadata: dict = {}
    if mode is not None:
        metadata["trajectory_mode"] = mode
    if effect is not None:
        metadata["effect_size"] = effect
    if anchor:
        metadata["zero_effect_anchor"] = True
    return SimulationReplicateResult(
        cell_id=cell_id,
        phase=phase,
        replicate_index=index,
        replicate_seed=index,
        generator_seed=index,
        evaluation_seed=index,
        parameter_signature="sig",
        status="completed",
        p_values={"delta": 0.01, "angle": 0.01, "shape": 0.01},
        pair_statistics={"delta": 1.0, "angle": 0.5, "shape": 0.25},
        realized_geometry={
            "schema_version": 1,
            "requested": {"trajectory_mode": mode, "group_effect_size": effect},
            "checkpoints": checkpoints,
        },
        truth_metadata={},
        runtime_metadata={"runtime_seconds": 0.1},
        cell_metadata=metadata,
        integration_metadata={"integration_method": "pls"},
        attribution_status="not_requested",
        attribution_diagnostics={},
    )


def _joint_only_records() -> list[SimulationReplicateResult]:
    """A mode whose response exists only once the blocks are concatenated."""

    flat = {
        "population_standardized": {
            "methylation": _scope(9.0, 0.0, 0.0),
            "expression": _scope(0.0, 0.0, 0.0),
            "joint": _scope(6.0, 20.0, 0.02),
        },
        "pls_latent": {"joint": _scope(6.0, 7.0, 0.02)},
    }
    anchor = {
        "population_standardized": {
            "methylation": _scope(0.0, 0.0, 0.0),
            "expression": _scope(0.0, 0.0, 0.0),
            "joint": _scope(0.0, 0.0, 0.0),
        },
        "pls_latent": {"joint": _scope(0.0, 0.0, 0.0)},
    }
    return [
        _record("mag", mode="magnitude", effect=1.0, checkpoints=flat),
        _record("anchor", mode="none", effect=0.0, checkpoints=anchor, anchor=True),
    ]


def test_decompose_surfaces_both_block_and_joint_scopes():
    frame = decompose_block_response(_joint_only_records(), mode="magnitude")
    standardized = frame[frame["checkpoint"] == "population_standardized"]
    assert set(standardized["scope"]) == {"methylation", "expression", "joint"}
    assert standardized[standardized["scope"] == "joint"]["is_joint"].all()
    assert not standardized[standardized["scope"] == "methylation"]["is_joint"].any()


def test_decompose_excludes_the_anchor_from_mode_rows_but_reports_its_value():
    frame = decompose_block_response(_joint_only_records(), mode="magnitude")
    # the anchor is a reference column, never a row of its own
    assert set(frame["trajectory_mode"]) == {"magnitude"}
    joint = frame[
        (frame["checkpoint"] == "population_standardized")
        & (frame["scope"] == "joint")
        & (frame["statistic"] == "angle")
    ]
    assert joint["anchor_median"].iloc[0] == pytest.approx(0.0)
    assert joint["excess_over_anchor"].iloc[0] == pytest.approx(20.0)


def test_missing_scope_is_absent_not_zero_filled():
    """``pls_latent`` records only the joint scope; 0 would be a real value."""

    frame = decompose_block_response(_joint_only_records(), mode="magnitude")
    latent = frame[frame["checkpoint"] == "pls_latent"]
    assert set(latent["scope"]) == {"joint"}
    assert "methylation" not in set(latent["scope"])


def test_joint_only_flag_identifies_a_concatenation_artifact():
    frame = decompose_block_response(_joint_only_records(), mode="magnitude")
    summary = summarize_block_localization(frame)
    row = summary[
        (summary["checkpoint"] == "population_standardized")
        & (summary["statistic"] == "angle")
    ].iloc[0]
    assert row["joint_only"]
    assert row["max_block_response"] == pytest.approx(0.0)
    assert row["joint_response"] == pytest.approx(20.0)

    # delta genuinely moves inside methylation, so it is not a joint artifact
    delta_row = summary[
        (summary["checkpoint"] == "population_standardized")
        & (summary["statistic"] == "delta")
    ].iloc[0]
    assert not delta_row["joint_only"]


def test_machine_precision_zeros_count_as_flat():
    """Procrustes/eigen dust must not read as a real per-block response.

    Regression guard: an exact ``== 0`` flatness test would classify the very
    constructions that *are* per-block pure as impure.
    """

    records = _joint_only_records()
    dusty = {
        "population_standardized": {
            "methylation": _scope(9.0, 0.0, 8.4e-18),
            "expression": _scope(0.0, 0.0, 1.4e-17),
            "joint": _scope(6.0, 20.0, 0.02),
        },
    }
    records[0] = _record("mag", mode="magnitude", effect=1.0, checkpoints=dusty)
    summary = summarize_block_localization(
        decompose_block_response(records, mode="magnitude")
    )
    shape_row = summary[
        (summary["checkpoint"] == "population_standardized")
        & (summary["statistic"] == "shape")
    ].iloc[0]
    assert shape_row["joint_only"]


def test_within_block_response_is_not_joint_only():
    """A mode that rotates each block individually is a different diagnosis."""

    per_block = {
        "population_standardized": {
            "methylation": _scope(2.0, 90.0, 0.10),
            "expression": _scope(2.0, 88.0, 0.10),
            "joint": _scope(2.0, 89.0, 0.06),
        },
    }
    anchor = {
        "population_standardized": {
            "methylation": _scope(0.0, 0.0, 0.0),
            "expression": _scope(0.0, 0.0, 0.0),
            "joint": _scope(0.0, 0.0, 0.0),
        },
    }
    records = [
        _record("ori", mode="orientation", effect=1.0, checkpoints=per_block),
        _record("anchor", mode="none", effect=0.0, checkpoints=anchor, anchor=True),
    ]
    summary = summarize_block_localization(
        decompose_block_response(records, mode="orientation")
    )
    assert not summary["joint_only"].any()


def test_checkpoint_without_a_joint_scope_is_never_joint_only():
    native = {
        "population_native": {
            "methylation": _scope(9.0, 0.0, 0.0),
            "expression": _scope(0.0, 0.0, 0.0),
        },
    }
    anchor = {
        "population_native": {
            "methylation": _scope(0.0, 0.0, 0.0),
            "expression": _scope(0.0, 0.0, 0.0),
        },
    }
    records = [
        _record("mag", mode="magnitude", effect=1.0, checkpoints=native),
        _record("anchor", mode="none", effect=0.0, checkpoints=anchor, anchor=True),
    ]
    summary = summarize_block_localization(
        decompose_block_response(records, mode="magnitude")
    )
    assert not summary["joint_only"].any()


def test_unknown_mode_and_empty_records_are_refused():
    with pytest.raises(ValueError, match="No non-anchor records"):
        decompose_block_response(_joint_only_records(), mode="orientation")
    with pytest.raises(ValueError, match="No replicate records"):
        decompose_block_response([], mode="magnitude")


def test_summary_of_an_empty_frame_has_the_expected_columns():
    import pandas as pd

    summary = summarize_block_localization(pd.DataFrame())
    assert list(summary.columns) == [
        "checkpoint",
        "statistic",
        "top_effect_size",
        "max_block_response",
        "joint_response",
        "joint_only",
    ]
