"""Crossed design grid: configuration, enumeration, seeds, and headroom.

The design grid crosses declared axes into design points and enumerates one
anchored power grid per point (`phase5-design-point-pilot`). These tests pin
the contract on small configs with the R-free `concat` evaluator.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from motco.simulations import SemiSyntheticTrajectoryParams, SimulationEvaluationParams
from motco.simulations.grid import derive_replicate_seed, parameter_signature
from motco.simulations.study import (
    AcceptanceTargets,
    DesignGrid,
    DesignPointDecisionRule,
    MatchedSeedPolicy,
    StudyConfig,
    StudyConfigError,
    dump_study_config,
    enumerate_study,
    load_study_config,
)
from motco.simulations.study.enumerate import (
    DESIGN_AXIS_MARKER,
    DESIGN_PHASE,
    DESIGN_POINT_KEY,
    SEED_FAMILY_KEY,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "examples" / "trajectory_power_study"
SNAPSHOT = Path(__file__).resolve().parent / "data" / "study_cell_identity_snapshot.json"

RHO = "generator.baseline_continuity"
N = "generator.n_samples"


def _config(**overrides) -> StudyConfig:
    defaults: dict = {
        "generator": SemiSyntheticTrajectoryParams(
            seed=2, trajectory_mode="magnitude", n_samples=60, p_dmp=0.1
        ),
        "evaluation": SimulationEvaluationParams(integration_method="concat", permutations=0, seed=3),
        "trajectory_modes": ("magnitude", "orientation"),
        "effect_sizes": (0.0, 0.5, 1.0),
        "n_replicates": 2,
        "base_seed": 100,
        "matched_seeds": MatchedSeedPolicy(enabled=True, primary_family="fam"),
        "design_grid": DesignGrid(axes={RHO: (0.0, 0.5), N: (60, 120)}),
    }
    defaults.update(overrides)
    return StudyConfig(**defaults)


def _rule(**overrides) -> DesignPointDecisionRule:
    defaults: dict = {
        "trajectory_mode": "orientation",
        "statistic": "angle",
        "min_power_at_top": 0.8,
        "prefer": (N, RHO),
    }
    defaults.update(overrides)
    return DesignPointDecisionRule(**defaults)


# ── Configuration ─────────────────────────────────────────────────────────────


def test_design_grid_requires_the_baseline_value_on_every_axis() -> None:
    with pytest.raises(StudyConfigError, match="must include the baseline value"):
        _config(design_grid=DesignGrid(axes={RHO: (0.5, 0.8), N: (60,)}))


def test_design_grid_rejects_unknown_namespace_and_empty_values() -> None:
    with pytest.raises(StudyConfigError, match="unsupported namespace"):
        DesignGrid(axes={"intersim.n_samples": (60,)})
    with pytest.raises(StudyConfigError, match="at least one value"):
        DesignGrid(axes={N: ()})
    with pytest.raises(StudyConfigError, match="unknown generator field"):
        _config(design_grid=DesignGrid(axes={"generator.not_a_field": (1,)}))


def test_design_grid_axis_cannot_also_be_an_ofat_axis() -> None:
    with pytest.raises(StudyConfigError, match="also declared under axes"):
        _config(axes={N: (60, 240)})


def test_design_point_rule_validation() -> None:
    with pytest.raises(StudyConfigError, match="unknown"):
        _rule(trajectory_mode="spiral")
    with pytest.raises(StudyConfigError, match="unknown"):
        _rule(statistic="volume")
    with pytest.raises(StudyConfigError, match="between 0 and 1"):
        _rule(min_power_at_top=1.5)
    with pytest.raises(StudyConfigError, match="at least one design axis"):
        _rule(prefer=())
    with pytest.raises(StudyConfigError, match="absent from design_grid.axes"):
        _config(acceptance=AcceptanceTargets(design_point=_rule(prefer=("generator.p_dmp",))))
    with pytest.raises(StudyConfigError, match="requires a non-empty design_grid"):
        _config(design_grid=DesignGrid(), acceptance=AcceptanceTargets(design_point=_rule()))
    with pytest.raises(StudyConfigError, match="absent from trajectory_modes"):
        _config(acceptance=AcceptanceTargets(design_point=_rule(trajectory_mode="shape")))


def test_design_grid_and_rule_round_trip_through_json(tmp_path: Path) -> None:
    config = _config(acceptance=AcceptanceTargets(design_point=_rule()))
    path = tmp_path / "design.json"
    dump_study_config(config, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["design_grid"]["axes"] == {RHO: [0.0, 0.5], N: [60, 120]}
    assert payload["acceptance"]["design_point"]["prefer"] == [N, RHO]
    assert load_study_config(path) == config


def test_configs_without_a_design_grid_dump_an_empty_grid_and_no_rule(tmp_path: Path) -> None:
    config = _config(design_grid=DesignGrid())
    path = tmp_path / "plain.json"
    dump_study_config(config, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["design_grid"] == {"axes": {}}
    assert payload["acceptance"]["design_point"] is None
    assert load_study_config(path) == config


def test_design_points_cross_axes_in_declared_order() -> None:
    config = _config()
    points = config.design_points()
    assert len(points) == 4
    assert points[0] == {RHO: 0.0, N: 60}
    assert config.baseline_design_point() == {RHO: 0.0, N: 60}
    assert config.is_baseline_design_point(points[0])
    assert not any(config.is_baseline_design_point(point) for point in points[1:])


# ── Enumeration ───────────────────────────────────────────────────────────────


def test_every_non_baseline_point_gets_an_anchor_and_the_nonzero_power_grid() -> None:
    config = _config()
    grid = enumerate_study(config)
    design = [cell for cell in grid.cells if cell.phase == DESIGN_PHASE]
    n_points = len(config.design_points()) - 1
    n_nonzero = len([e for e in config.effect_sizes if e != 0.0])
    assert len(design) == n_points * (1 + len(config.trajectory_modes) * n_nonzero)
    anchors = [cell for cell in design if cell.metadata.get("zero_effect_anchor")]
    assert len(anchors) == n_points
    for cell in design:
        assert cell.metadata["varied_axis"] == DESIGN_AXIS_MARKER
        assert set(cell.metadata[DESIGN_POINT_KEY]) == {RHO, N}
        assert not config.is_baseline_design_point(cell.metadata[DESIGN_POINT_KEY])
        point = cell.metadata[DESIGN_POINT_KEY]
        assert cell.generator_params.baseline_continuity == point[RHO]
        assert cell.generator_params.n_samples == point[N]
    for anchor in anchors:
        assert anchor.generator_params.trajectory_mode == "none"
        assert anchor.generator_params.group_effect_size == 0.0
        assert anchor.metadata["resolves_modes"] == list(config.trajectory_modes)
    assert grid.metadata["design_grid"]["n_points"] == 4


def test_baseline_column_is_the_primary_grid_stamped_with_its_point() -> None:
    grid = enumerate_study(_config())
    primary = [cell for cell in grid.cells if cell.phase == "power_primary"]
    assert len(primary) == 1 + 2 * 2
    for cell in primary:
        assert cell.metadata["varied_axis"] is None
        assert cell.metadata[DESIGN_POINT_KEY] == {RHO: 0.0, N: 60}


def test_design_cells_share_the_primary_seed_family() -> None:
    grid = enumerate_study(_config())
    primary = next(cell for cell in grid.cells if cell.phase == "power_primary")
    for cell in grid.cells:
        if cell.phase == DESIGN_PHASE:
            assert cell.metadata[SEED_FAMILY_KEY] == "fam"
            for index in range(2):
                assert derive_replicate_seed(cell, index) == derive_replicate_seed(primary, index)
        elif cell.phase.startswith("type_i"):
            assert cell.metadata[SEED_FAMILY_KEY] != "fam"


def test_duplicate_datasets_inside_the_family_are_still_rejected() -> None:
    # A design axis on an evaluation field keeps the datasets identical on
    # purpose; a generator axis that does not change the dataset is a duplicate.
    ok = _config(design_grid=DesignGrid(axes={"evaluation.permutations": (0, 9)}))
    assert any(cell.phase == DESIGN_PHASE for cell in enumerate_study(ok).cells)
    with pytest.raises(StudyConfigError, match="identical datasets"):
        enumerate_study(_config(design_grid=DesignGrid(axes={"generator.magnitude_kind": ("all", "extremes")})))


def test_headroom_is_enforced_at_each_design_point() -> None:
    # Baseline rho = 0.9 fits orientation at e = 1.0 with p_dmp = 0.2 (saturates
    # near 2.8); the rho = 0 design point does not (saturates near 0.56).
    config = _config(
        generator=SemiSyntheticTrajectoryParams(
            seed=2, trajectory_mode="magnitude", n_samples=60, p_dmp=0.2, baseline_continuity=0.9
        ),
        design_grid=DesignGrid(axes={RHO: (0.9, 0.0)}),
    )
    with pytest.raises(StudyConfigError) as excinfo:
        enumerate_study(config)
    message = str(excinfo.value)
    assert "power_design" in message
    assert "trajectory_mode='orientation'" in message
    assert "saturates at group_effect_size" in message
    # The same effect enumerates once the offending point is removed.
    enumerate_study(
        _config(
            generator=SemiSyntheticTrajectoryParams(
                seed=2, trajectory_mode="magnitude", n_samples=60, p_dmp=0.2, baseline_continuity=0.9
            ),
            design_grid=DesignGrid(axes={RHO: (0.9, 0.6)}),
        )
    )


def test_configs_without_a_design_grid_enumerate_no_design_cells() -> None:
    grid = enumerate_study(_config(design_grid=DesignGrid()))
    assert not any(cell.phase == DESIGN_PHASE for cell in grid.cells)
    assert not any(DESIGN_POINT_KEY in cell.metadata for cell in grid.cells)
    assert "design_grid" not in grid.metadata


def test_enumeration_is_deterministic() -> None:
    first = enumerate_study(_config())
    second = enumerate_study(_config())
    assert [c.cell_id for c in first.cells] == [c.cell_id for c in second.cells]
    assert [parameter_signature(c) for c in first.cells] == [parameter_signature(c) for c in second.cells]


@pytest.mark.parametrize("name", sorted(json.loads(SNAPSHOT.read_text(encoding="utf-8"))))
def test_committed_configs_keep_their_cell_ids_and_signatures(name: str) -> None:
    """Snapshot of every committed config's (cell id, parameter signature) pairs.

    Taken from the code revision before the design grid existed; a diff here
    means a study config that predates the feature would no longer resume.
    """

    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))[name]
    grid = enumerate_study(load_study_config(CONFIG_DIR / name))
    actual = sorted([cell.cell_id, parameter_signature(cell)] for cell in grid.cells)
    assert actual == expected
