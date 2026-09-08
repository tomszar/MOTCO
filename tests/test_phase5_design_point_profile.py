"""The committed Phase 5 design-point pilot profile enumerates as declared."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from motco.simulations.reference import load_reference
from motco.simulations.semisynthetic import expected_surgery_headroom
from motco.simulations.study import enumerate_study, load_study_config
from motco.simulations.study.enumerate import DESIGN_PHASE, DESIGN_POINT_KEY, SEED_FAMILY_KEY

CONFIG = Path("examples/trajectory_power_study/phase5_design_point_pilot.json")
RHO = "generator.baseline_continuity"
N = "generator.n_samples"


def test_profile_declares_the_design_point_pilot() -> None:
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert "surgery_censoring" not in raw["generator"], "new configs must not copy the clamp flag"
    assert raw["generator"]["p_dmp"] == 0.1
    assert raw["design_grid"]["axes"] == {RHO: [0.0, 0.5, 0.8], N: [300, 600, 1200]}
    assert raw["acceptance"]["gate"] == {"enabled": False}
    assert raw["attribution"] == {"enabled": False}
    assert raw["acceptance"]["design_point"]["prefer"] == [N, RHO]

    config = load_study_config(CONFIG)
    assert config.generator.surgery_censoring == "error"
    assert config.evaluation.integration_method == "pls"
    assert config.evaluation.permutations == 199
    assert config.evaluation.n_jobs == 1
    assert config.n_replicates == 100
    assert config.trajectory_modes == ("orientation", "translation")
    assert config.effect_sizes == (0.0, 0.25, 0.5, 1.0)
    assert config.acceptance.design_point is not None
    assert config.acceptance.design_point.min_power_at_top == 0.8
    assert not config.attribution.enabled
    assert not config.acceptance.gate.enabled


def test_profile_enumerates_nine_columns_without_censoring() -> None:
    config = load_study_config(CONFIG)
    grid = enumerate_study(config)  # raises on any over-headroom cell
    phases = Counter(cell.phase for cell in grid.cells)
    # Baseline column: shared anchor + 2 modes × 3 nonzero effects.
    assert phases["power_primary"] == 1 + 2 * 3
    # Eight further design points, each with its own anchor and power grid.
    assert phases[DESIGN_PHASE] == 8 * (1 + 2 * 3)
    assert phases["type_i_baseline"] == 2
    assert sum(cell.n_replicates for cell in grid.cells) == 100 * len(grid.cells)

    points = {
        json.dumps(cell.metadata[DESIGN_POINT_KEY], sort_keys=True)
        for cell in grid.cells
        if DESIGN_POINT_KEY in cell.metadata
    }
    assert len(points) == 9
    family = config.matched_seeds.primary_family
    for cell in grid.cells:
        if cell.phase in ("power_primary", DESIGN_PHASE):
            assert cell.metadata[SEED_FAMILY_KEY] == family
            assert cell.generator_params.surgery_censoring == "error"

    reference = load_reference()
    for cell in grid.cells:
        headroom = expected_surgery_headroom(cell.generator_params, reference=reference)
        assert headroom is None or headroom.fits, cell.cell_id
