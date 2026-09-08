"""Nested evaluation axes: ``evaluation.integration_params.<key>`` with ``null`` = absent.

Covers the grid helpers, the study-config validators and baseline resolution,
the parameter-signature contract (design D2 of
``resolve-latent-rank-at-design-point``), and the enumeration invariants of a
rank-only design grid: every column shares generator identity and replicate
seeds with the baseline (same data, different measurement) and the
duplicate-dataset guard still fires on truly identical cells.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from motco.simulations import SemiSyntheticTrajectoryParams, SimulationEvaluationParams
from motco.simulations.grid import (
    SimulationGridError,
    _apply_axis_value,
    _get_axis_value,
    _split_axis,
    derive_replicate_seed,
    make_simulation_cell,
    parameter_signature,
)
from motco.simulations.study import (
    AcceptanceTargets,
    DesignGrid,
    MatchedSeedPolicy,
    RankDecisionRule,
    StatisticPair,
    StudyConfig,
    StudyConfigError,
    TypeIBound,
    dump_study_config,
    enumerate_study,
    load_study_config,
)
from motco.simulations.study.enumerate import (
    DESIGN_PHASE,
    DESIGN_POINT_KEY,
    SEED_FAMILY_KEY,
    _evaluation_identity,
    _generator_identity,
)

RANK = "evaluation.integration_params.forced_components"


def _generator(**overrides) -> SemiSyntheticTrajectoryParams:
    defaults = {"seed": 2, "trajectory_mode": "magnitude", "n_samples": 60, "p_dmp": 0.1}
    defaults.update(overrides)
    return SemiSyntheticTrajectoryParams(**defaults)


def _evaluation(**overrides) -> SimulationEvaluationParams:
    defaults = {
        "integration_method": "pls",
        "integration_params": {"cv1_splits": 3, "cv2_splits": 4, "random_state": 1203},
        "permutations": 0,
        "seed": 3,
    }
    defaults.update(overrides)
    return SimulationEvaluationParams(**defaults)


# ── grid helpers ──────────────────────────────────────────────────────────────


def test_split_axis_accepts_top_level_and_the_one_nested_form() -> None:
    assert _split_axis("generator.n_samples") == ("generator", ("n_samples",))
    assert _split_axis("evaluation.permutations") == ("evaluation", ("permutations",))
    assert _split_axis(RANK) == ("evaluation", ("integration_params", "forced_components"))


@pytest.mark.parametrize(
    "axis",
    [
        "generator.a.b",
        "evaluation.attribution.x",
        "evaluation.integration_params.a.b",
        "generator.integration_params.forced_components",
    ],
)
def test_split_axis_rejects_other_nesting_by_name(axis: str) -> None:
    with pytest.raises(SimulationGridError, match=axis.replace(".", r"\.")):
        _split_axis(axis)


def test_nested_axis_value_is_applied_beside_existing_knobs() -> None:
    generator, evaluation = _apply_axis_value(_generator(), _evaluation(), RANK, 6)
    assert generator == _generator()
    assert dict(evaluation.integration_params) == {
        "cv1_splits": 3,
        "cv2_splits": 4,
        "random_state": 1203,
        "forced_components": 6,
    }
    assert _get_axis_value(generator, evaluation, RANK) == 6
    # The original mapping is untouched.
    assert "forced_components" not in _evaluation().integration_params


def test_none_removes_the_nested_key_and_reads_back_as_none() -> None:
    _, forced = _apply_axis_value(_generator(), _evaluation(), RANK, 6)
    _, cleared = _apply_axis_value(_generator(), forced, RANK, None)
    assert "forced_components" not in cleared.integration_params
    assert cleared == _evaluation()
    assert _get_axis_value(_generator(), _evaluation(), RANK) is None
    assert _get_axis_value(_generator(), None, RANK) is None


def test_top_level_axes_behave_as_before() -> None:
    generator, evaluation = _apply_axis_value(_generator(), _evaluation(), "generator.n_samples", 120)
    assert generator.n_samples == 120 and evaluation == _evaluation()
    generator, evaluation = _apply_axis_value(_generator(), _evaluation(), "evaluation.permutations", 9)
    assert evaluation.permutations == 9 and generator == _generator()
    assert _get_axis_value(_generator(), _evaluation(), "generator.n_samples") == 60


def test_nested_axis_values_are_distinguished_by_the_parameter_signature() -> None:
    def cell(value):
        generator, evaluation = _apply_axis_value(_generator(), _evaluation(), RANK, value)
        return make_simulation_cell(
            phase="power_design", generator_params=generator, evaluation_params=evaluation, base_seed=1
        )

    plain = make_simulation_cell(
        phase="power_design", generator_params=_generator(), evaluation_params=_evaluation(), base_seed=1
    )
    assert parameter_signature(cell(None)) != parameter_signature(cell(6))
    assert parameter_signature(cell(6)) != parameter_signature(cell(9))
    # D2: applying null is byte-identical to never declaring the axis.
    assert parameter_signature(cell(None)) == parameter_signature(plain)
    assert cell(None).cell_id == plain.cell_id


# ── study config ──────────────────────────────────────────────────────────────


def _config(**overrides) -> StudyConfig:
    defaults: dict = {
        "generator": _generator(),
        "evaluation": _evaluation(),
        "trajectory_modes": ("magnitude", "orientation", "shape", "translation"),
        "effect_sizes": (0.0, 0.5, 1.0),
        "n_replicates": 2,
        "base_seed": 100,
        "matched_seeds": MatchedSeedPolicy(enabled=True, primary_family="ladder"),
        "design_grid": DesignGrid(axes={RANK: (None, 4, 6)}),
    }
    defaults.update(overrides)
    return StudyConfig(**defaults)


def _rule(**overrides) -> RankDecisionRule:
    defaults: dict = {
        "axis": RANK,
        "target": StatisticPair("orientation", "angle"),
        "protected": (StatisticPair("magnitude", "delta"), StatisticPair("shape", "shape")),
        "type_i_bound": TypeIBound(alpha=0.05, se_tolerance=2.0),
        "gain_se_multiplier": 2.0,
        "loss_se_multiplier": 2.0,
    }
    defaults.update(overrides)
    return RankDecisionRule(**defaults)


def test_rank_axis_loads_with_a_null_baseline() -> None:
    config = _config()
    assert config.axis_baseline_value(RANK) is None
    assert config.baseline_design_point() == {RANK: None}
    assert config.is_baseline_design_point({RANK: None})
    assert not config.is_baseline_design_point({RANK: 4})
    # A config that already forces a rank resolves the baseline from the mapping.
    forced = _config(
        evaluation=_evaluation(integration_params={"forced_components": 4}),
        design_grid=DesignGrid(axes={RANK: (4, 6)}),
    )
    assert forced.axis_baseline_value(RANK) == 4


def test_rank_axis_without_null_is_rejected_naming_the_axis_and_the_missing_value() -> None:
    with pytest.raises(StudyConfigError, match=r"forced_components.*baseline value None"):
        _config(design_grid=DesignGrid(axes={RANK: (4, 6)}))


@pytest.mark.parametrize("axis", ["generator.a.b", "evaluation.attribution.x"])
def test_config_rejects_other_nesting(axis: str) -> None:
    with pytest.raises(StudyConfigError, match="only nested form supported"):
        DesignGrid(axes={axis: (1,)})
    with pytest.raises(StudyConfigError, match="only nested form supported"):
        _config(design_grid=DesignGrid(), axes={axis: (1,)})


def test_null_survives_dump_and_reload(tmp_path: Path) -> None:
    config = _config(acceptance=AcceptanceTargets(rank_decision=_rule()))
    path = tmp_path / "ladder.json"
    dump_study_config(config, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["design_grid"]["axes"] == {RANK: [None, 4, 6]}
    assert payload["acceptance"]["rank_decision"]["axis"] == RANK
    assert payload["acceptance"]["rank_decision"]["protected"] == [
        {"trajectory_mode": "magnitude", "statistic": "delta"},
        {"trajectory_mode": "shape", "statistic": "shape"},
    ]
    reloaded = load_study_config(path)
    assert reloaded == config
    assert reloaded.acceptance.rank_decision == _rule()


def test_rank_decision_rule_validation() -> None:
    _config(acceptance=AcceptanceTargets(rank_decision=_rule()))
    with pytest.raises(StudyConfigError, match="not declared under design_grid.axes"):
        _config(design_grid=DesignGrid(), acceptance=AcceptanceTargets(rank_decision=_rule()))
    with pytest.raises(StudyConfigError, match="not declared under design_grid.axes"):
        _config(
            design_grid=DesignGrid(axes={"generator.n_samples": (60, 120)}),
            acceptance=AcceptanceTargets(rank_decision=_rule()),
        )
    with pytest.raises(StudyConfigError, match="unknown"):
        _rule(protected=(StatisticPair("spiral", "delta"),))
    with pytest.raises(StudyConfigError, match="absent from trajectory_modes"):
        _config(
            trajectory_modes=("orientation", "magnitude"),
            acceptance=AcceptanceTargets(rank_decision=_rule()),
        )
    with pytest.raises(StudyConfigError, match="non-negative"):
        _rule(gain_se_multiplier=-1.0)
    with pytest.raises(StudyConfigError, match="non-negative"):
        _rule(loss_se_multiplier=-0.5)
    with pytest.raises(StudyConfigError, match="at least one pair"):
        _rule(protected=())
    with pytest.raises(StudyConfigError, match="cannot also be protected"):
        _rule(protected=(StatisticPair("orientation", "angle"),))
    with pytest.raises(StudyConfigError, match="nested evaluation integration parameter"):
        _rule(axis="generator.n_samples")


def test_rank_decision_rule_loads_from_json_and_rejects_unknown_fields(tmp_path: Path) -> None:
    raw = {
        "generator": {"seed": 2, "trajectory_mode": "magnitude", "n_samples": 60, "p_dmp": 0.1},
        "evaluation": {"integration_method": "pls", "permutations": 0, "seed": 3},
        "trajectory_modes": ["magnitude", "orientation", "shape", "translation"],
        "effect_sizes": [0.0, 0.5, 1.0],
        "design_grid": {"axes": {RANK: [None, 4, 6]}},
        "matched_seeds": {"enabled": True, "primary_family": "ladder"},
        "acceptance": {
            "rank_decision": {
                "axis": RANK,
                "target": {"trajectory_mode": "orientation", "statistic": "angle"},
                "protected": [
                    {"trajectory_mode": "magnitude", "statistic": "delta"},
                    {"trajectory_mode": "shape", "statistic": "shape"},
                ],
                "type_i_bound": {"alpha": 0.05, "se_tolerance": 2.0},
                "gain_se_multiplier": 1.5,
                "loss_se_multiplier": 2.5,
            }
        },
    }
    path = tmp_path / "rule.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    config = load_study_config(path)
    rule = config.acceptance.rank_decision
    assert rule is not None
    assert rule.gain_se_multiplier == 1.5 and rule.loss_se_multiplier == 2.5
    assert rule.target == StatisticPair("orientation", "angle")

    raw["acceptance"]["rank_decision"]["bogus"] = 1
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(StudyConfigError, match="unknown field"):
        load_study_config(path)


# ── enumeration invariants (design D3) ────────────────────────────────────────


def test_rank_columns_share_generator_identity_and_seeds_with_the_baseline() -> None:
    config = _config()
    grid = enumerate_study(config)  # no duplicate-dataset error, no headroom rejection
    primary = {
        (cell.metadata["trajectory_mode"], cell.metadata["effect_size"]): cell
        for cell in grid.cells
        if cell.phase == "power_primary"
    }
    design = [cell for cell in grid.cells if cell.phase == DESIGN_PHASE]
    n_nonzero = len([e for e in config.effect_sizes if e != 0.0])
    assert len(design) == 2 * (1 + len(config.trajectory_modes) * n_nonzero)
    assert {cell.metadata[DESIGN_POINT_KEY][RANK] for cell in design} == {4, 6}
    for cell in design:
        twin = primary[(cell.metadata["trajectory_mode"], cell.metadata["effect_size"])]
        assert cell.metadata[SEED_FAMILY_KEY] == twin.metadata[SEED_FAMILY_KEY] == "ladder"
        assert _generator_identity(cell) == _generator_identity(twin)
        assert cell.generator_params == twin.generator_params
        assert _evaluation_identity(cell) != _evaluation_identity(twin)
        for index in range(config.n_replicates):
            assert derive_replicate_seed(cell, index) == derive_replicate_seed(twin, index)
        rank = cell.metadata[DESIGN_POINT_KEY][RANK]
        assert cell.evaluation_params.integration_params["forced_components"] == rank
        assert parameter_signature(cell) != parameter_signature(twin)
    for cell in primary.values():
        assert "forced_components" not in cell.evaluation_params.integration_params
        assert cell.metadata[DESIGN_POINT_KEY] == {RANK: None}


def test_identical_generator_and_evaluation_identity_is_still_rejected() -> None:
    # A generator axis that does not change the dataset collides with the
    # baseline on both identities and must still be refused.
    with pytest.raises(StudyConfigError, match="identical datasets"):
        enumerate_study(
            _config(design_grid=DesignGrid(axes={"generator.magnitude_kind": ("all", "extremes")}))
        )


def test_rank_axis_as_ofat_axis_enumerates_forced_cells() -> None:
    grid = enumerate_study(_config(design_grid=DesignGrid(), axes={RANK: (None, 5)}))
    ofat = [cell for cell in grid.cells if cell.phase == "power_ofat"]
    assert ofat and all(cell.evaluation_params.integration_params["forced_components"] == 5 for cell in ofat)
    assert all(cell.metadata["varied_axis"] == RANK and cell.metadata["varied_value"] == 5 for cell in ofat)
    type_i = [cell for cell in grid.cells if cell.phase == "type_i_ofat"]
    assert len(type_i) == 1 and type_i[0].evaluation_params.integration_params["forced_components"] == 5
