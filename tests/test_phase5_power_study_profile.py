"""The committed Phase 5 paper-grade profile enumerates as declared."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from motco.simulations.grid import derive_replicate_seed
from motco.simulations.reference import load_reference
from motco.simulations.semisynthetic import expected_surgery_headroom
from motco.simulations.study import enumerate_study, load_study_config
from motco.simulations.study.enumerate import SEED_FAMILY_KEY

CONFIG = Path("examples/trajectory_power_study/phase5_power_study.json")
LADDER = Path("examples/trajectory_power_study/phase5_latent_rank_ladder.json")
TEMPLATE = Path("examples/trajectory_power_study/phase5_report_template.md")


def test_profile_is_the_ladder_cv_column_at_paper_grade_precision() -> None:
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    ladder = json.loads(LADDER.read_text(encoding="utf-8"))

    assert raw["metadata"]["derives_from"] == str(LADDER)
    assert raw["generator"] == ladder["generator"], "generator must be copied from the ladder"
    assert raw["evaluation"]["integration_params"] == ladder["evaluation"]["integration_params"]
    assert raw["evaluation"]["integration_method"] == "pls"
    assert "design_grid" not in raw
    assert "surgery_censoring" not in raw["generator"], "new configs must not copy the clamp flag"
    assert raw["effect_sizes"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert raw["report_contract"] == {
        "driver_component": "observed",
        "cross_replicate_driver_agreement": "descriptive",
        "n_jobs_override": "forbid",
    }

    config = load_study_config(CONFIG)
    assert config.generator.surgery_censoring == "error"
    assert config.generator.n_samples == 1200
    assert config.generator.n_stages == 4
    assert config.generator.p_dmp == 0.1
    assert config.generator.baseline_continuity == 0.0
    assert "forced_components" not in config.evaluation.integration_params
    assert config.evaluation.permutations == 999
    assert config.evaluation.n_jobs == 1
    assert config.n_replicates == 500
    assert config.alpha == 0.05
    assert config.trajectory_modes == ("magnitude", "orientation", "shape", "translation")
    assert not config.axes and not config.design_grid.enabled
    assert config.matched_seeds.enabled and config.matched_seeds.shared_zero_effect_anchor
    assert config.matched_seeds.primary_family == "phase5-primary"
    assert config.matched_seeds.primary_family not in {"phase4-primary", "phase5-design-point", "phase5-latent-rank"}
    assert config.base_seed not in {ladder["base_seed"]}

    contract = config.report_contract
    assert contract is not None
    assert contract.driver_component == "observed"
    assert contract.cross_replicate_driver_agreement == "descriptive"
    assert contract.forbids_n_jobs_override


def test_profile_attribution_gate_and_targets_are_the_phase4_roles() -> None:
    config = load_study_config(CONFIG)

    attribution = config.attribution
    assert attribution.enabled
    assert attribution.trajectory_modes == ("orientation",)
    assert attribution.phases == ("power_primary",)
    assert attribution.nonzero_effects_only and attribution.effect_sizes is None
    assert (attribution.bootstrap_replicates, attribution.top_k, attribution.bootstrap_seed) == (100, 20, 0)

    gate = config.acceptance.gate
    assert gate.enabled
    assert gate.control_modes == ("none", "translation")
    roles = {(rule.trajectory_mode, rule.statistic): rule.role for rule in gate.rules}
    assert roles == {
        ("magnitude", "delta"): "mandatory_power",
        ("orientation", "angle"): "mandatory_power",
        ("shape", "shape"): "mandatory_power",
        ("magnitude", "angle"): "mandatory_control",
        ("magnitude", "shape"): "mandatory_control",
        ("orientation", "delta"): "descriptive",
        ("orientation", "shape"): "descriptive",
        ("shape", "delta"): "descriptive",
        ("shape", "angle"): "descriptive",
    }
    assert gate.min_power_at_top == 0.8

    acceptance = config.acceptance
    assert [(t.alpha, t.se_tolerance) for t in acceptance.type_i] == [(0.05, 2.0)]
    assert {(t.trajectory_mode, t.statistic, t.min_power_at_top) for t in acceptance.power} == {
        ("magnitude", "delta", 0.8),
        ("orientation", "angle", 0.8),
        ("shape", "shape", 0.8),
    }
    specificity = {(t.trajectory_mode, t.statistic) for t in acceptance.specificity}
    mandatory_controls = {pair for pair, role in roles.items() if role == "mandatory_control"}
    assert specificity == mandatory_controls | {("translation", s) for s in ("delta", "angle", "shape")}
    assert ("orientation", "shape") not in specificity, "predeclared cross-talk is not a target"
    assert ("shape", "delta") not in specificity and ("shape", "angle") not in specificity
    assert acceptance.design_point is None and acceptance.rank_decision is None


def test_profile_enumerates_nineteen_cells_without_censoring() -> None:
    config = load_study_config(CONFIG)
    grid = enumerate_study(config)  # raises on any over-headroom or duplicate-dataset cell
    phases = Counter(cell.phase for cell in grid.cells)
    assert phases == {"type_i_baseline": 2, "power_primary": 1 + 4 * 4}
    assert len(grid.cells) == 19
    assert sum(cell.n_replicates for cell in grid.cells) == 9_500

    anchors = [cell for cell in grid.cells if cell.metadata.get("zero_effect_anchor")]
    assert len(anchors) == 1
    assert anchors[0].metadata["resolves_modes"] == list(config.trajectory_modes)
    assert anchors[0].generator_params.group_effect_size == 0.0

    primary = [cell for cell in grid.cells if cell.phase == "power_primary"]
    assert {cell.metadata[SEED_FAMILY_KEY] for cell in primary} == {"phase5-primary"}
    assert all(cell.generator_params.surgery_censoring == "error" for cell in grid.cells)
    assert all(cell.evaluation_params.permutations == 999 for cell in grid.cells)
    assert all(cell.evaluation_params.n_jobs == 1 for cell in grid.cells)
    # Matched seeds: every primary cell shares the anchor's generator seed at each replicate index.
    for cell in primary:
        for index in (0, 1, 499):
            assert derive_replicate_seed(cell, index) == derive_replicate_seed(anchors[0], index)
    # Attribution is armed on exactly the nonzero orientation cells.
    armed = {
        (cell.metadata["trajectory_mode"], cell.metadata["effect_size"])
        for cell in primary
        if cell.evaluation_params.attribution.enabled
    }
    assert armed == {("orientation", e) for e in (0.25, 0.5, 0.75, 1.0)}

    reference = load_reference()
    for cell in grid.cells:
        headroom = expected_surgery_headroom(cell.generator_params, reference=reference)
        assert headroom is None or headroom.fits, cell.cell_id


def test_profile_needs_no_r_runtime() -> None:
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert "intersim" not in json.dumps(raw).lower()


def test_report_template_names_every_contract_item() -> None:
    text = TEMPLATE.read_text(encoding="utf-8")
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    contract = raw["report_contract"]
    for heading in (
        "## 1. Configuration and provenance",
        "## 2. Unit and failure accounting",
        "## 3. Gate decision",
        "## 4. Type I error",
        "## 5. Power per mode",
        "## 6. Cross-talk",
        "## 7. Drivers",
        "## 8. Construction limitations",
        "## 9. Reproduction",
    ):
        assert heading in text, heading
    provenance_fields = (
        "config_sha256", "code_revision", "versions", "unit_timings", "shard layout", "error_policy", "n_jobs"
    )
    for field in provenance_fields:
        assert field in text, field
    assert f"`{contract['driver_component']}`" in text
    assert "no cross-replicate driver-stability claim" in text.lower()
    assert contract["cross_replicate_driver_agreement"] in text
    assert "counted as one measurement" in text
    assert "eigengap" in text and "null width" in text.lower()
    assert "not ρ-invariant" in text or "not rho-invariant" in text
    assert "n-conditional" in text
    assert "`phase4_`" in text or "phase4_*" in text
    reproduction = text[text.index("## 9. Reproduction"):]
    assert "--n-jobs" not in reproduction.replace("no `--n-jobs`", "").replace("without `--n-jobs`", "")
    assert "report_contract.json" in text and "driver_report.csv" in text
