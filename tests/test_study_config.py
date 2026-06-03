from __future__ import annotations

import json
from pathlib import Path

import pytest

from motco.simulations import SemiSyntheticTrajectoryParams, SimulationEvaluationParams
from motco.simulations.study import (
    AcceptanceTargets,
    PowerMonotonicityTarget,
    SpecificityTarget,
    StudyConfig,
    StudyConfigError,
    TypeIControlTarget,
    enumerate_study,
    load_study_config,
)
from motco.simulations.study.enumerate import NEGATIVE_CONTROL_MODES


def _baseline_config(**overrides) -> StudyConfig:
    defaults: dict = {
        "generator": SemiSyntheticTrajectoryParams(seed=2, trajectory_mode="magnitude", n_samples=60),
        "evaluation": SimulationEvaluationParams(integration_method="concat", permutations=0, seed=3),
        "trajectory_modes": ("magnitude", "orientation", "shape"),
        "effect_sizes": (0.1, 0.5),
        "axes": {"generator.n_samples": (60, 120)},
        "n_replicates": 2,
        "base_seed": 100,
        "alpha": 0.05,
        "acceptance": AcceptanceTargets(
            type_i=(TypeIControlTarget(alpha=0.05),),
            power=(PowerMonotonicityTarget(trajectory_mode="magnitude", statistic="delta", min_power_at_top=0.8),),
            specificity=(SpecificityTarget(trajectory_mode="translation", statistic="angle", alpha=0.05),),
        ),
    }
    defaults.update(overrides)
    return StudyConfig(**defaults)


def test_enumerate_study_is_deterministic_and_negative_controls_present() -> None:
    config = _baseline_config()
    first = enumerate_study(config)
    second = enumerate_study(config)

    assert [c.cell_id for c in first.cells] == [c.cell_id for c in second.cells]

    null_cells = [c for c in first.cells if c.phase.startswith("type_i_")]
    null_modes = {c.generator_params.trajectory_mode for c in null_cells}
    for mode in NEGATIVE_CONTROL_MODES:
        assert mode in null_modes, f"negative-control mode {mode!r} missing from Type I cells"


def test_enumerate_study_unique_cell_ids() -> None:
    grid = enumerate_study(_baseline_config())
    ids = [c.cell_id for c in grid.cells]
    assert len(ids) == len(set(ids))


def test_invalid_axes_namespace_is_rejected() -> None:
    with pytest.raises(StudyConfigError, match="namespace"):
        _baseline_config(axes={"intersim.n_sample": (20, 40)})


def test_unknown_trajectory_mode_is_rejected() -> None:
    with pytest.raises(StudyConfigError, match="Unknown trajectory mode"):
        _baseline_config(trajectory_modes=("not_a_mode",))


def test_negative_n_replicates_is_rejected() -> None:
    with pytest.raises(StudyConfigError, match="non-negative"):
        _baseline_config(n_replicates=-1)


def test_empty_trajectory_modes_or_effect_sizes_is_rejected() -> None:
    with pytest.raises(StudyConfigError, match="trajectory_modes"):
        _baseline_config(trajectory_modes=())
    with pytest.raises(StudyConfigError, match="effect_sizes"):
        _baseline_config(effect_sizes=())


def test_alpha_out_of_range_is_rejected() -> None:
    with pytest.raises(StudyConfigError, match="alpha"):
        _baseline_config(alpha=0.0)
    with pytest.raises(StudyConfigError, match="alpha"):
        _baseline_config(alpha=1.0)


def test_load_study_config_json(tmp_path: Path) -> None:
    path = tmp_path / "study.json"
    payload = {
        "generator": {"seed": 2, "trajectory_mode": "magnitude", "group_ratio": 0.5, "n_samples": 60},
        "evaluation": {"integration_method": "concat", "permutations": 0, "seed": 3},
        "trajectory_modes": ["magnitude", "translation"],
        "effect_sizes": [0.1, 0.5],
        "axes": {"generator.n_samples": [60, 120]},
        "n_replicates": 2,
        "base_seed": 1,
        "alpha": 0.05,
        "acceptance": {
            "type_i": [{"alpha": 0.05, "se_tolerance": 2.0}],
            "power": [{"trajectory_mode": "magnitude", "statistic": "delta", "min_power_at_top": 0.8}],
            "specificity": [{"trajectory_mode": "translation", "statistic": "angle", "alpha": 0.05}],
        },
        "metadata": {"study": "smoke"},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    config = load_study_config(path)
    assert config.generator.seed == 2
    assert config.generator.trajectory_mode == "magnitude"
    assert config.trajectory_modes == ("magnitude", "translation")
    assert config.effect_sizes == (0.1, 0.5)
    assert config.axes["generator.n_samples"] == (60, 120)
    assert len(config.acceptance.type_i) == 1
    assert config.acceptance.power[0].trajectory_mode == "magnitude"
    assert config.metadata["study"] == "smoke"


def test_load_study_config_missing_required_field(tmp_path: Path) -> None:
    path = tmp_path / "study.json"
    path.write_text(json.dumps({"trajectory_modes": ["magnitude"], "effect_sizes": [0.1]}), encoding="utf-8")
    with pytest.raises(StudyConfigError, match="missing required field"):
        load_study_config(path)


def test_load_study_config_unknown_generator_field(tmp_path: Path) -> None:
    path = tmp_path / "study.json"
    payload = {
        "generator": {"seed": 2, "not_a_field": 7},
        "evaluation": {},
        "trajectory_modes": ["magnitude"],
        "effect_sizes": [0.1],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StudyConfigError, match="unknown field"):
        load_study_config(path)
