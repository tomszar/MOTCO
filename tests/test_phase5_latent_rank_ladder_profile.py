"""The committed Phase 5 latent-rank ladder profile enumerates as declared."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from motco.simulations.grid import derive_replicate_seed
from motco.simulations.reference import load_reference
from motco.simulations.semisynthetic import expected_surgery_headroom
from motco.simulations.study import enumerate_study, load_study_config
from motco.simulations.study.enumerate import (
    DESIGN_PHASE,
    DESIGN_POINT_KEY,
    SEED_FAMILY_KEY,
    _generator_identity,
)
from motco.simulations.study.spectrum import RANK_AXIS

CONFIG = Path("examples/trajectory_power_study/phase5_latent_rank_ladder.json")
PILOT = Path("examples/trajectory_power_study/phase5_design_point_pilot.json")


def test_profile_declares_the_latent_rank_ladder() -> None:
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert "surgery_censoring" not in raw["generator"], "new configs must not copy the clamp flag"
    assert raw["generator"]["p_dmp"] == 0.1
    assert raw["generator"]["n_samples"] == 1200
    assert raw["generator"]["baseline_continuity"] == 0.0
    assert raw["design_grid"]["axes"] == {RANK_AXIS: [None, 3, 4, 6, 9, 12]}
    assert raw["acceptance"]["gate"] == {"enabled": False}
    assert raw["attribution"] == {"enabled": False}
    assert raw["acceptance"]["rank_decision"]["target"] == {"trajectory_mode": "orientation", "statistic": "angle"}
    assert raw["metadata"]["derives_from"] == str(PILOT)

    pilot = json.loads(PILOT.read_text(encoding="utf-8"))
    assert {**pilot["generator"], "n_samples": 1200} == raw["generator"]
    assert pilot["evaluation"] == raw["evaluation"]

    config = load_study_config(CONFIG)
    assert config.generator.surgery_censoring == "error"
    assert config.evaluation.integration_method == "pls"
    assert config.evaluation.permutations == 199
    assert config.evaluation.n_jobs == 1
    assert "forced_components" not in config.evaluation.integration_params
    assert config.n_replicates == 100
    assert config.trajectory_modes == ("magnitude", "orientation", "shape", "translation")
    assert config.effect_sizes == (0.0, 0.25, 0.5, 1.0)
    assert config.matched_seeds.enabled and config.matched_seeds.shared_zero_effect_anchor
    assert config.matched_seeds.primary_family == "phase5-latent-rank"
    assert config.axis_baseline_value(RANK_AXIS) is None
    rule = config.acceptance.rank_decision
    assert rule is not None
    assert rule.axis == RANK_AXIS
    assert [pair.label for pair in rule.protected] == ["magnitude/delta", "shape/shape"]
    assert rule.gain_se_multiplier == 2.0 and rule.loss_se_multiplier == 2.0
    assert rule.type_i_bound.alpha == 0.05 and rule.type_i_bound.se_tolerance == 2.0
    assert config.acceptance.type_i and not config.attribution.enabled and not config.acceptance.gate.enabled
    assert config.acceptance.design_point is None


def test_profile_enumerates_six_paired_columns_without_censoring() -> None:
    config = load_study_config(CONFIG)
    grid = enumerate_study(config)  # raises on any over-headroom or duplicate-dataset cell
    phases = Counter(cell.phase for cell in grid.cells)
    # Baseline (CV) column: shared anchor + 4 modes × 3 nonzero effects.
    assert phases["power_primary"] == 1 + 4 * 3
    # Five forced columns, each with its own anchor and power grid → 78 power cells in total.
    assert phases[DESIGN_PHASE] == 5 * (1 + 4 * 3)
    assert phases["power_primary"] + phases[DESIGN_PHASE] == 78
    assert phases["type_i_baseline"] == 2
    assert len(grid.cells) == 80
    assert sum(cell.n_replicates for cell in grid.cells) == 8000

    columns = {cell.metadata[DESIGN_POINT_KEY][RANK_AXIS] for cell in grid.cells if DESIGN_POINT_KEY in cell.metadata}
    assert columns == {None, 3, 4, 6, 9, 12}

    primary = {
        (cell.metadata["trajectory_mode"], cell.metadata["effect_size"]): cell
        for cell in grid.cells
        if cell.phase == "power_primary"
    }
    family = config.matched_seeds.primary_family
    for cell in grid.cells:
        if cell.phase not in ("power_primary", DESIGN_PHASE):
            continue
        assert cell.metadata[SEED_FAMILY_KEY] == family
        assert cell.generator_params.surgery_censoring == "error"
        rank = cell.metadata[DESIGN_POINT_KEY][RANK_AXIS]
        params = dict(cell.evaluation_params.integration_params)
        if rank is None:
            assert "forced_components" not in params
        else:
            assert params["forced_components"] == rank
        twin = primary[(cell.metadata["trajectory_mode"], cell.metadata["effect_size"])]
        assert _generator_identity(cell) == _generator_identity(twin)
        assert cell.generator_params == twin.generator_params
        for index in (0, 1, 99):
            assert derive_replicate_seed(cell, index) == derive_replicate_seed(twin, index)

    reference = load_reference()
    for cell in grid.cells:
        headroom = expected_surgery_headroom(cell.generator_params, reference=reference)
        assert headroom is None or headroom.fits, cell.cell_id
