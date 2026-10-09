"""Explicit group × stage sample sizes: generator validation, sizing, and study configs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from motco.simulations import (
    SemiSyntheticTrajectoryError,
    SemiSyntheticTrajectoryParams,
    SimulationEvaluationParams,
    generate_semisynthetic_trajectory,
    load_reference,
)
from motco.simulations.grid import derive_replicate_seed, parameter_signature
from motco.simulations.study import (
    DesignGrid,
    MatchedSeedPolicy,
    StudyConfig,
    enumerate_study,
    load_study_config,
)
from motco.simulations.study.enumerate import DESIGN_PHASE, DESIGN_POINT_KEY

SEA_AD = ((11, 10, 28), (9, 10, 12))
SEA_AD_50 = ((10, 9, 25), (9, 9, 12))
SIZES = "generator.group_stage_sizes"
LAYERS = "evaluation.integration_params.layers"


@pytest.fixture(scope="module")
def reference():
    return load_reference()


def _params(**overrides) -> SemiSyntheticTrajectoryParams:
    base: dict = {"seed": 3, "trajectory_mode": "magnitude", "group_effect_size": 0.5, "group_stage_sizes": SEA_AD}
    base.update(overrides)
    return SemiSyntheticTrajectoryParams(**base)


# ── Generator ────────────────────────────────────────────────────────────────


def test_explicit_sizes_are_realized_exactly(reference) -> None:
    dataset = generate_semisynthetic_trajectory(_params(), reference=reference)
    counts = dataset.metadata.groupby(["group", "stage"]).size()
    assert [int(counts[("A", stage)]) for stage in range(3)] == [11, 10, 28]
    assert [int(counts[("B", stage)]) for stage in range(3)] == [9, 10, 12]
    assert len(dataset.metadata) == 80
    for layer in ("methylation", "expression", "proteomics"):
        assert getattr(dataset, layer).shape[0] == 80
    assert dataset.truth["group_stage_sizes"] == {"A": [11, 10, 28], "B": [9, 10, 12]}


def test_absent_table_records_no_sizes_in_truth(reference) -> None:
    dataset = generate_semisynthetic_trajectory(
        SemiSyntheticTrajectoryParams(seed=3, n_samples=60), reference=reference
    )
    assert "group_stage_sizes" not in dataset.truth


@pytest.mark.parametrize(
    ("override", "named"),
    [
        ({"n_samples": 80}, "n_samples"),
        ({"stage_sample_prop": (0.3, 0.3, 0.4)}, "stage_sample_prop"),
        ({"group_ratio": 0.6}, "group_ratio"),
    ],
)
def test_table_conflicts_with_non_default_proportional_settings(reference, override, named) -> None:
    with pytest.raises(SemiSyntheticTrajectoryError, match=named):
        generate_semisynthetic_trajectory(_params(**override), reference=reference)


def test_conflict_error_names_every_conflicting_setting(reference) -> None:
    with pytest.raises(SemiSyntheticTrajectoryError, match="n_samples, group_ratio"):
        generate_semisynthetic_trajectory(_params(n_samples=80, group_ratio=0.6), reference=reference)


@pytest.mark.parametrize(
    "table",
    [
        ((11, 10, 28),),
        ((11, 10, 28), (9, 10, 12), (1, 1, 1)),
        ((11, 10), (9, 10)),
        ((11, 10, 28), (9, 10, 12, 4)),
        ((11, 0, 28), (9, 10, 12)),
        ((11, 10, 28), (9, -1, 12)),
    ],
    ids=["one-row", "three-rows", "short-rows", "long-row", "zero-cell", "negative-cell"],
)
def test_malformed_table_is_rejected(reference, table) -> None:
    with pytest.raises(SemiSyntheticTrajectoryError, match="group_stage_sizes"):
        generate_semisynthetic_trajectory(_params(group_stage_sizes=table), reference=reference)


def test_same_seed_gives_same_dataset_with_a_table(reference) -> None:
    first = generate_semisynthetic_trajectory(_params(), reference=reference)
    second = generate_semisynthetic_trajectory(_params(), reference=reference)
    assert first.metadata.equals(second.metadata)
    assert first.methylation.equals(second.methylation)


# ── Study configuration ──────────────────────────────────────────────────────


def _config(**overrides) -> StudyConfig:
    defaults: dict = {
        "generator": SemiSyntheticTrajectoryParams(
            seed=2, trajectory_mode="magnitude", p_dmp=0.1, group_stage_sizes=SEA_AD
        ),
        "evaluation": SimulationEvaluationParams(
            integration_method="concat",
            integration_params={"layers": ["methylation", "expression"]},
            permutations=0,
            seed=3,
        ),
        "trajectory_modes": ("magnitude", "orientation"),
        "effect_sizes": (0.0, 0.5),
        "n_replicates": 2,
        "base_seed": 100,
        "matched_seeds": MatchedSeedPolicy(enabled=True, primary_family="fam"),
    }
    defaults.update(overrides)
    return StudyConfig(**defaults)


def _write_config(tmp_path: Path, generator_sizes, design_axes=None) -> Path:
    payload = {
        "generator": {
            "seed": 2,
            "trajectory_mode": "magnitude",
            "p_dmp": 0.1,
            "magnitude_kind": "joint",
            "group_stage_sizes": generator_sizes,
        },
        "evaluation": {"integration_method": "concat", "permutations": 0, "seed": 3},
        "trajectory_modes": ["magnitude", "orientation"],
        "effect_sizes": [0.0, 0.5],
        "n_replicates": 2,
        "base_seed": 100,
        "matched_seeds": {"enabled": True, "primary_family": "fam"},
    }
    if design_axes is not None:
        payload["design_grid"] = {"axes": design_axes}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _identity(config: StudyConfig) -> dict[str, tuple[str, list[int]]]:
    return {
        cell.cell_id: (parameter_signature(cell), [derive_replicate_seed(cell, i) for i in range(cell.n_replicates)])
        for cell in enumerate_study(config).cells
    }


def test_size_table_round_trips_through_a_config(tmp_path: Path) -> None:
    path = _write_config(tmp_path, [[11, 10, 28], [9, 10, 12]])
    config = load_study_config(path)
    assert config.generator.group_stage_sizes == SEA_AD
    assert _identity(config) == _identity(load_study_config(path))
    # The in-process (tuple) spelling hashes like the JSON (list) spelling.
    assert _identity(config) == _identity(_config(evaluation=config.evaluation))
    cell = next(cell for cell in enumerate_study(config).cells if cell.phase == "power_primary")
    dataset = generate_semisynthetic_trajectory(cell.generator_params)
    assert len(dataset.metadata) == 80


def test_size_table_design_axis_enumerates_one_anchored_grid_per_table(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        [[11, 10, 28], [9, 10, 12]],
        design_axes={SIZES: [[[11, 10, 28], [9, 10, 12]], [[10, 9, 25], [9, 9, 12]]]},
    )
    config = load_study_config(path)
    assert config.design_grid.axes[SIZES] == (SEA_AD, SEA_AD_50)
    design = [cell for cell in enumerate_study(config).cells if cell.phase == DESIGN_PHASE]
    assert len(design) == 1 + 2 * 1
    assert [cell for cell in design if cell.metadata.get("zero_effect_anchor")]
    for cell in design:
        assert cell.generator_params.group_stage_sizes == SEA_AD_50
        assert tuple(tuple(row) for row in cell.metadata[DESIGN_POINT_KEY][SIZES]) == SEA_AD_50


def test_layer_columns_share_generator_params_and_seeds() -> None:
    config = _config(design_grid=DesignGrid(axes={LAYERS: (["methylation", "expression"], None)}))
    cells = enumerate_study(config).cells
    primary = {
        (cell.generator_params.trajectory_mode, cell.generator_params.group_effect_size): cell
        for cell in cells
        if cell.phase == "power_primary"
    }
    design = [cell for cell in cells if cell.phase == DESIGN_PHASE]
    assert design
    for cell in design:
        assert "layers" not in cell.evaluation_params.integration_params
        partner = primary[(cell.generator_params.trajectory_mode, cell.generator_params.group_effect_size)]
        assert cell.generator_params == partner.generator_params
        for index in range(cell.n_replicates):
            assert derive_replicate_seed(cell, index) == derive_replicate_seed(partner, index)
