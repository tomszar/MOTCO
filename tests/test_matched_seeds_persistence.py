"""Matched generator seeds, signature versioning, and Phase 4 persistence."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from motco.simulations import (
    AttributionDiagnosticSettings,
    SemiSyntheticTrajectoryParams,
    SimulationEvaluationParams,
    SimulationEvaluationResult,
)
from motco.simulations.attribution_diagnostics import ATTRIBUTION_SCHEMA_VERSION
from motco.simulations.evaluation import INTEGRATION_METADATA_VERSION
from motco.simulations.grid import (
    SEED_DERIVATION_VERSION,
    SEED_FAMILY_KEY,
    SimulationGridError,
    SimulationRunConfig,
    append_replicate_results,
    derive_replicate_seed,
    make_simulation_cell,
    parameter_signature,
    read_replicate_results,
    run_simulation_grid,
    run_simulation_replicate,
)
from motco.simulations.grid import SimulationGrid as Grid
from motco.simulations.study import (
    MatchedSeedPolicy,
    StudyConfig,
    enumerate_study,
)

# These tests exercise seed derivation and cell identity, not the effect axis:
# they opt into clamping so the top effects stay enumerable without the grid
# having to be reshaped around the surgery headroom.
_GENERATOR = SemiSyntheticTrajectoryParams(
    seed=2, trajectory_mode="magnitude", n_samples=60, n_stages=4, surgery_censoring="clamp"
)
_PLS_EVALUATION = SimulationEvaluationParams(integration_method="pls", permutations=0, seed=3)


def _config(**overrides) -> StudyConfig:
    defaults: dict = {
        "generator": _GENERATOR,
        "evaluation": _PLS_EVALUATION,
        "trajectory_modes": ("magnitude", "orientation", "shape", "translation"),
        "effect_sizes": (0.0, 0.5, 1.0),
        "n_replicates": 3,
        "base_seed": 100,
        "matched_seeds": MatchedSeedPolicy(enabled=True),
    }
    defaults.update(overrides)
    return StudyConfig(**defaults)


def _cell(cell_id: str, *, metadata: dict | None = None, n_replicates: int = 3):
    return make_simulation_cell(
        phase="power_primary",
        generator_params=_GENERATOR,
        evaluation_params=_PLS_EVALUATION,
        n_replicates=n_replicates,
        base_seed=100,
        metadata=metadata or {},
        cell_id=cell_id,
    )


def _primary(grid) -> list:
    return [c for c in grid.cells if c.phase == "power_primary"]


# --- 4.1 matched seeds --------------------------------------------------------


def test_matched_seeds_are_shared_across_mode_and_effect_cells() -> None:
    grid = enumerate_study(_config())
    primary = _primary(grid)
    assert len({c.metadata[SEED_FAMILY_KEY] for c in primary}) == 1

    for replicate_index in range(3):
        seeds = {derive_replicate_seed(cell, replicate_index) for cell in primary}
        assert len(seeds) == 1, "primary cells must share one seed at each replicate index"


def test_seeds_differ_across_replicate_indices() -> None:
    grid = enumerate_study(_config())
    cell = _primary(grid)[0]
    seeds = [derive_replicate_seed(cell, index) for index in range(3)]
    assert len(set(seeds)) == 3


def test_negative_controls_keep_independent_seed_families() -> None:
    grid = enumerate_study(_config())
    controls = [c for c in grid.cells if c.phase != "power_primary"]
    assert controls
    families = {c.metadata[SEED_FAMILY_KEY] for c in controls}
    assert len(families) == len(controls), "each control needs its own family"

    primary_family = _primary(grid)[0].metadata[SEED_FAMILY_KEY]
    assert primary_family not in families
    primary_seed = derive_replicate_seed(_primary(grid)[0], 0)
    assert all(derive_replicate_seed(c, 0) != primary_seed for c in controls)


def test_legacy_behavior_is_preserved_without_the_policy() -> None:
    """A cell with no seed family keeps the pre-Phase-4 derivation payload."""

    plain = _cell("legacy-cell")
    assert SEED_FAMILY_KEY not in plain.metadata
    assert derive_replicate_seed(plain, 0) == derive_replicate_seed(_cell("legacy-cell"), 0)
    assert derive_replicate_seed(plain, 0) != derive_replicate_seed(_cell("other-cell"), 0)

    grid = enumerate_study(_config(matched_seeds=MatchedSeedPolicy()))
    assert all(SEED_FAMILY_KEY not in c.metadata for c in grid.cells)
    seeds = {derive_replicate_seed(c, 0) for c in _primary(grid)}
    assert len(seeds) == len(_primary(grid)), "without the policy every cell draws independently"


def test_persistence_keys_stay_unique_under_matched_seeds() -> None:
    grid = enumerate_study(_config())
    keys = [(c.cell_id, index) for c in grid.cells for index in range(c.n_replicates)]
    assert len(keys) == len(set(keys))


def test_seed_family_change_changes_the_signature() -> None:
    base = _cell("cell", metadata={SEED_FAMILY_KEY: "primary"})
    other = _cell("cell", metadata={SEED_FAMILY_KEY: "control:cell"})
    assert parameter_signature(base) != parameter_signature(other)


# --- 4.1a / 4.6 shared zero-effect anchor -------------------------------------


def test_single_zero_effect_anchor_is_emitted() -> None:
    grid = enumerate_study(_config())
    anchors = [c for c in _primary(grid) if c.metadata.get("zero_effect_anchor")]
    assert len(anchors) == 1
    anchor = anchors[0]
    assert anchor.metadata["effect_size"] == 0.0
    assert anchor.generator_params.group_effect_size == 0.0
    assert set(anchor.metadata["resolves_modes"]) == set(_config().trajectory_modes)
    assert anchor.metadata[SEED_FAMILY_KEY] == _primary(grid)[0].metadata[SEED_FAMILY_KEY]

    zero_cells = [c for c in _primary(grid) if float(c.metadata["effect_size"]) == 0.0]
    assert zero_cells == anchors, "no per-mode zero-effect duplicates may be enumerated"


def test_every_mode_resolves_its_zero_point_to_the_anchor() -> None:
    config = _config()
    grid = enumerate_study(config)
    anchor = next(c for c in _primary(grid) if c.metadata.get("zero_effect_anchor"))
    for mode in config.trajectory_modes:
        assert mode in anchor.metadata["resolves_modes"]
        nonzero = {
            float(c.metadata["effect_size"])
            for c in _primary(grid)
            if c.metadata.get("trajectory_mode") == mode and not c.metadata.get("zero_effect_anchor")
        }
        assert 0.0 not in nonzero
        assert nonzero == {0.5, 1.0}


def test_no_two_primary_cells_generate_identical_datasets() -> None:
    grid = enumerate_study(_config())
    primary = _primary(grid)
    identities = set()
    for cell in primary:
        params = replace(cell.generator_params, seed=derive_replicate_seed(cell, 0))
        if float(params.group_effect_size) == 0.0:
            # At a zero requested effect the generator ignores the mode entirely.
            params = replace(params, trajectory_mode="none", shape_kind="relocate", magnitude_kind="all")
        identity = tuple(sorted(vars(params).items()))
        assert identity not in identities, f"{cell.cell_id} duplicates another primary cell's dataset"
        identities.add(identity)
    assert len(identities) == len(primary)


def test_duplicate_primary_datasets_are_rejected() -> None:
    """Two identical primary cells in one family must not survive enumeration."""

    from motco.simulations.study.enumerate import _require_unique_ids

    duplicate = [
        _cell("a", metadata={SEED_FAMILY_KEY: "primary", "effect_size": 1.0}),
        _cell("b", metadata={SEED_FAMILY_KEY: "primary", "effect_size": 1.0}),
    ]
    with pytest.raises(Exception, match="identical datasets"):
        _require_unique_ids(duplicate)


# --- 4.2 signature versions ---------------------------------------------------


def test_signature_carries_every_diagnostic_schema_version() -> None:
    cell = _cell("cell")
    payload = json.loads(_signature_payload(cell))
    assert payload["seed_derivation_version"] == SEED_DERIVATION_VERSION
    assert payload["realized_geometry_version"] >= 1
    assert payload["integration_metadata_version"] == INTEGRATION_METADATA_VERSION
    assert payload["attribution_schema_version"] == ATTRIBUTION_SCHEMA_VERSION


def test_attribution_settings_reach_the_signature_through_evaluation_params() -> None:
    base = _cell("cell")
    enabled = make_simulation_cell(
        phase="power_primary",
        generator_params=_GENERATOR,
        evaluation_params=replace(
            _PLS_EVALUATION, attribution=AttributionDiagnosticSettings(enabled=True, bootstrap_replicates=5)
        ),
        n_replicates=3,
        base_seed=100,
        cell_id="cell",
    )
    assert parameter_signature(base) != parameter_signature(enabled)


def _signature_payload(cell) -> str:
    from motco.simulations.grid import _to_jsonable

    payload = {
        "phase": cell.phase,
        "generator_params": _to_jsonable(cell.generator_params),
        "evaluation_params": _to_jsonable(cell.evaluation_params),
        "n_replicates": cell.n_replicates,
        "base_seed": cell.base_seed,
        "metadata": _to_jsonable(cell.metadata),
        "seed_derivation_version": SEED_DERIVATION_VERSION,
        "realized_geometry_version": 1,
        "integration_metadata_version": INTEGRATION_METADATA_VERSION,
        "attribution_schema_version": ATTRIBUTION_SCHEMA_VERSION,
    }
    return json.dumps(payload)


# --- 4.3 / 4.4 persistence ----------------------------------------------------


def _fake_result(*, attribution: dict | None = None) -> SimulationEvaluationResult:
    return SimulationEvaluationResult(
        observed_deltas=np.zeros((2, 2)),
        observed_angles=np.zeros((2, 2)),
        observed_shapes=np.zeros((2, 2)),
        pair_statistics={"delta": 1.0, "angle": 0.5, "shape": 0.25},
        p_values={"delta": 0.01, "angle": 0.2, "shape": 0.4},
        latent_matrix_metadata={
            "integration_method": "pls",
            "integration_metadata_version": INTEGRATION_METADATA_VERSION,
            "selected_lv": 4,
            "cv_mean_auroc": 0.87,
            "integration_params": {"cv1_splits": 3, "cv2_splits": 4},
            "feature_order_signature": "abc123",
        },
        truth_metadata={"trajectory_mode": "orientation"},
        runtime_metadata={"runtime_seconds": 0.5},
        evaluation_params=_PLS_EVALUATION,
        group_levels=["A", "B"],
        stage_levels=["0", "1", "2", "3"],
        contrast=[[0, 1, 2, 3], [4, 5, 6, 7]],
        realized_geometry={"schema_version": 1, "checkpoints": {}},
        attribution_diagnostics=attribution
        if attribution is not None
        else {"schema_version": ATTRIBUTION_SCHEMA_VERSION, "status": "not_requested", "reason": "n/a"},
    )


def _evaluator(attribution: dict | None = None):
    def run(generator_params, evaluation_params):
        return _fake_result(attribution=attribution)

    return run


def test_completed_record_carries_integration_and_attribution(tmp_path: Path) -> None:
    attribution = {
        "schema_version": ATTRIBUTION_SCHEMA_VERSION,
        "status": "computed",
        "settings": {"top_k": 20, "bootstrap_replicates": 100},
        "model": {"selected_components": 4},
        "top_features": [{"feature": "methylation__cg0", "rank": 1, "effect_standardized": 0.5}],
    }
    cell = _cell("cell", n_replicates=1)
    record = run_simulation_replicate(cell, 0, evaluator=_evaluator(attribution))

    assert record.integration_metadata["selected_lv"] == 4
    assert record.attribution_status == "computed"
    assert record.attribution_diagnostics["top_features"][0]["rank"] == 1
    assert record.diagnostic_error_message is None

    path = tmp_path / "out.jsonl"
    append_replicate_results(path, [record])
    loaded = read_replicate_results(path)[0]
    assert loaded == record


def test_ineligible_cell_is_marked_not_requested(tmp_path: Path) -> None:
    cell = _cell("cell", n_replicates=1)
    record = run_simulation_replicate(cell, 0, evaluator=_evaluator())
    assert record.attribution_status == "not_requested"
    assert record.diagnostic_error_type is None
    assert record.diagnostic_error_message is None


def test_diagnostic_failure_is_recorded_without_failing_the_replicate() -> None:
    failure = {
        "schema_version": ATTRIBUTION_SCHEMA_VERSION,
        "status": "failed",
        "reason": "Orientation attribution failed: missing complete group-by-stage cells",
        "error_type": "AttributionDiagnosticError",
    }
    record = run_simulation_replicate(_cell("cell", n_replicates=1), 0, evaluator=_evaluator(failure))
    assert record.status == "completed"
    assert record.attribution_status == "failed"
    assert record.diagnostic_error_type == "AttributionDiagnosticError"
    assert "missing complete group-by-stage cells" in record.diagnostic_error_message


def test_legacy_records_load_with_empty_additive_fields(tmp_path: Path) -> None:
    path = tmp_path / "legacy.jsonl"
    legacy = {
        "cell_id": "old",
        "phase": "power_primary",
        "replicate_index": 0,
        "replicate_seed": 1,
        "generator_seed": 1,
        "evaluation_seed": 2,
        "parameter_signature": "old-signature",
        "status": "completed",
        "p_values": {"delta": 0.01},
        "pair_statistics": {"delta": 1.0},
        "realized_geometry": {},
        "truth_metadata": {},
        "runtime_metadata": {},
        "cell_metadata": {},
    }
    path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")
    record = read_replicate_results(path)[0]

    assert record.integration_metadata == {}
    assert record.attribution_diagnostics == {}
    assert record.attribution_status == "not_requested"
    assert record.p_values == {"delta": 0.01}


def test_legacy_signature_cannot_satisfy_a_phase_4_resume(tmp_path: Path) -> None:
    cell = _cell("cell", n_replicates=1)
    path = tmp_path / "resume.jsonl"
    stale = run_simulation_replicate(cell, 0, evaluator=_evaluator())
    append_replicate_results(path, [replace(stale, parameter_signature="legacy-signature")])

    with pytest.raises(SimulationGridError, match="different\n?\\s*parameter signature"):
        run_simulation_grid(
            Grid(cells=(cell,)),
            config=SimulationRunConfig(output_path=path, resume=True),
            evaluator=_evaluator(),
        )


def test_matching_signature_resumes_without_rerunning(tmp_path: Path) -> None:
    cell = _cell("cell", n_replicates=1)
    path = tmp_path / "resume.jsonl"
    run_simulation_grid(
        Grid(cells=(cell,)),
        config=SimulationRunConfig(output_path=path, resume=True),
        evaluator=_evaluator(),
    )
    again = run_simulation_grid(
        Grid(cells=(cell,)),
        config=SimulationRunConfig(output_path=path, resume=True),
        evaluator=_evaluator(),
    )
    assert again == []


def test_persisted_record_is_json_safe_and_compact(tmp_path: Path) -> None:
    """A representative Phase 4 record must stay small enough for JSONL scale."""

    attribution = {
        "schema_version": ATTRIBUTION_SCHEMA_VERSION,
        "status": "computed",
        "settings": {"bootstrap_replicates": 100, "bootstrap_seed": 0, "top_k": 20, "zero_tolerance": 1e-12},
        "model": {"selected_components": 6, "feature_order_signature": "0123456789abcdef", "n_features": 1000},
        "units": {"standardized": "pooled_standardized_input", "methylation_units": "mvalue"},
        "truth": {"available": True, "n_drivers": 400},
        "transitions": [
            {
                "transition_id": f"{i}->{i + 1}",
                "components": {
                    name: {"path_length_group_1": 1.0, "path_length_group_2": 1.0, "contrast_norm": 0.5}
                    for name in ("observed", "pls_captured", "residual")
                },
                "retention": {"available": True, "cosine": 0.9, "norm_ratio": 0.8},
            }
            for i in range(3)
        ],
        "top_features": [
            {
                "transition_id": f"{i}->{i + 1}",
                "component": component,
                "rank": rank,
                "feature": f"methylation__cg{rank:05d}",
                "sign": 1,
                "effect_standardized": 0.5,
                "effect_original": 1.0,
                "sign_stability": 0.9,
                "top_k_selection_frequency": 0.7,
            }
            for i in range(3)
            for component in ("observed", "pls_captured", "residual")
            for rank in range(1, 21)
        ],
        "truth_recovery": [
            {"transition_id": f"{i}->{i + 1}", "component": "observed", "precision": 0.5, "recall": 0.02}
            for i in range(3)
        ],
        "stability": [
            {"transition_id": f"{i}->{i + 1}", "component": "observed", "mean_sign_stability": 0.8}
            for i in range(3)
        ],
        "runtime": {"attribution_seconds": 12.5},
    }
    record = run_simulation_replicate(_cell("cell", n_replicates=1), 0, evaluator=_evaluator(attribution))
    path = tmp_path / "record.jsonl"
    append_replicate_results(path, [record])

    line = path.read_text(encoding="utf-8").strip()
    json.loads(line)  # JSON-safe: no NaN, no Inf, no estimator
    assert len(line) < 100_000, f"representative record is {len(line)} bytes"
    assert read_replicate_results(path)[0] == record
