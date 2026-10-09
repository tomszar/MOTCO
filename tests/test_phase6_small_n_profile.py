"""The committed Phase 6 small-n pilot profile reproduces the SEA-AD case-study design."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from motco.simulations.grid import derive_replicate_seed, run_simulation_replicate
from motco.simulations.semisynthetic import generate_semisynthetic_trajectory
from motco.simulations.study import enumerate_study, load_study_config

PILOT = Path(__file__).resolve().parents[1] / "examples" / "trajectory_power_study" / "phase6_small_n_pilot.json"
PILOT_GRID = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0)
SEA_AD = ((11, 10, 28), (9, 10, 12))
MEASURED = ["methylation", "expression"]


def test_pilot_declares_the_case_study_design() -> None:
    raw = json.loads(PILOT.read_text(encoding="utf-8"))
    for proportional in ("n_samples", "stage_sample_prop", "group_ratio", "surgery_censoring"):
        assert proportional not in raw["generator"]
    assert "design_grid" not in raw and raw["axes"] == {}

    config = load_study_config(PILOT)
    assert config.generator.n_stages == 3
    assert config.generator.group_stage_sizes == SEA_AD
    assert config.generator.magnitude_kind == "joint"
    assert config.generator.surgery_censoring == "error"
    assert config.evaluation.integration_method == "pls"
    assert config.evaluation.integration_params["layers"] == MEASURED
    assert "forced_components" not in config.evaluation.integration_params
    assert config.trajectory_modes == ("magnitude", "orientation", "shape", "translation")
    assert config.effect_sizes == PILOT_GRID
    assert (config.n_replicates, config.evaluation.permutations) == (100, 199)
    assert config.matched_seeds.enabled and config.matched_seeds.shared_zero_effect_anchor
    assert config.matched_seeds.primary_family == "phase6-small-n"
    assert not config.acceptance.gate.enabled


def test_pilot_enumerates_under_headroom_with_the_declared_cell_sizes() -> None:
    config = load_study_config(PILOT)
    grid = enumerate_study(config)  # rejects any over-headroom cell
    assert len(grid.cells) == 2 + 1 + 4 * 8
    anchors = [cell for cell in grid.cells if cell.metadata.get("zero_effect_anchor")]
    assert len(anchors) == 1
    cell = grid.cells[0]
    params = replace(cell.generator_params, seed=derive_replicate_seed(cell, 0))
    counts = generate_semisynthetic_trajectory(params).metadata.groupby(["group", "stage"]).size()
    assert [[int(counts[(g, s)]) for s in range(3)] for g in ("A", "B")] == [list(row) for row in SEA_AD]


@pytest.mark.slow
def test_every_pilot_cell_runs_end_to_end() -> None:
    """One replicate of every cell at a few permutations, including CV at the 9-sample cells."""

    for cell in enumerate_study(load_study_config(PILOT)).cells:
        fast = replace(cell, evaluation_params=replace(cell.evaluation_params, permutations=9))
        record = run_simulation_replicate(fast, 0)
        assert record.status == "completed", (cell.cell_id, record.error_message)
        integration = record.integration_metadata
        assert integration["component_selection"] == "cv"
        assert integration["integration_params"]["layers"] == MEASURED
        assert integration["layer_feature_counts"] == {"methylation": 367, "expression": 131}
        assert record.truth_metadata["group_stage_sizes"] == {"A": [11, 10, 28], "B": [9, 10, 12]}
        assert set(record.p_values) == {"delta", "angle", "shape"}
