"""Phase 4 configuration and enumeration of attribution diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from motco.simulations import (
    AttributionDiagnosticSettings,
    SemiSyntheticTrajectoryParams,
    SimulationEvaluationParams,
)
from motco.simulations.evaluation import SimulationEvaluationError, _validate_evaluation_params
from motco.simulations.grid import parameter_signature
from motco.simulations.study import (
    AttributionSelector,
    GateRule,
    MatchedSeedPolicy,
    Phase4GateConfig,
    StudyConfig,
    StudyConfigError,
    enumerate_study,
    load_study_config,
)

_PLS_EVALUATION = SimulationEvaluationParams(integration_method="pls", permutations=0, seed=3)


def _config(**overrides) -> StudyConfig:
    defaults: dict = {
        "generator": SemiSyntheticTrajectoryParams(seed=2, trajectory_mode="magnitude", n_samples=60),
        "evaluation": _PLS_EVALUATION,
        "trajectory_modes": ("magnitude", "orientation", "shape", "translation"),
        "effect_sizes": (0.0, 0.5, 1.0),
        "n_replicates": 2,
        "base_seed": 100,
    }
    defaults.update(overrides)
    return StudyConfig(**defaults)


def _gate(**overrides) -> Phase4GateConfig:
    defaults: dict = {
        "enabled": True,
        "alpha": 0.05,
        "control_se_tolerance": 2.0,
        "monotonicity_se_tolerance": 2.0,
        "min_power_at_top": 0.8,
        "confirmation_se_threshold": 1.0,
        "rules": (GateRule(trajectory_mode="magnitude", statistic="delta", role="mandatory_power"),),
    }
    defaults.update(overrides)
    return Phase4GateConfig(**defaults)


# --- disabled defaults ------------------------------------------------------


def test_attribution_is_disabled_by_default() -> None:
    settings = SimulationEvaluationParams().attribution
    assert settings == AttributionDiagnosticSettings()
    assert settings.enabled is False
    assert settings.bootstrap_replicates == 0


def test_study_defaults_leave_attribution_and_matched_seeds_off() -> None:
    config = _config()
    assert config.attribution.enabled is False
    assert config.matched_seeds.enabled is False
    for cell in enumerate_study(config).cells:
        assert cell.evaluation_params.attribution.enabled is False
        assert "seed_family" not in cell.metadata


# --- valid nonzero-orientation selection ------------------------------------


def test_selector_enables_only_nonzero_primary_orientation_cells() -> None:
    selector = AttributionSelector(enabled=True, bootstrap_replicates=7, top_k=3)
    grid = enumerate_study(_config(attribution=selector))

    enabled = {
        (cell.metadata.get("trajectory_mode"), cell.metadata.get("effect_size"))
        for cell in grid.cells
        if cell.evaluation_params.attribution.enabled
    }
    assert enabled == {("orientation", 0.5), ("orientation", 1.0)}

    for cell in grid.cells:
        if not cell.evaluation_params.attribution.enabled:
            continue
        assert cell.phase == "power_primary"
        assert cell.evaluation_params.attribution.bootstrap_replicates == 7
        assert cell.evaluation_params.attribution.top_k == 3


def test_selection_does_not_depend_on_observed_p_values() -> None:
    """Eligibility is fixed at enumeration, so no p-value can influence it."""

    selector = AttributionSelector(enabled=True)
    grid = enumerate_study(_config(attribution=selector))
    eligible = [c for c in grid.cells if c.evaluation_params.attribution.enabled]
    assert eligible, "expected at least one eligible orientation cell"
    assert all(c.metadata.get("effect_size") != 0.0 for c in eligible)


def test_explicit_effect_sizes_narrow_the_selection() -> None:
    selector = AttributionSelector(enabled=True, effect_sizes=(1.0,))
    grid = enumerate_study(_config(attribution=selector))
    enabled = {
        (cell.metadata.get("trajectory_mode"), cell.metadata.get("effect_size"))
        for cell in grid.cells
        if cell.evaluation_params.attribution.enabled
    }
    assert enabled == {("orientation", 1.0)}


# --- invalid selectors ------------------------------------------------------


def test_selector_rejects_unknown_mode() -> None:
    with pytest.raises(StudyConfigError, match="unknown mode"):
        AttributionSelector(enabled=True, trajectory_modes=("bend",))


def test_selector_rejects_non_positive_top_k() -> None:
    with pytest.raises(StudyConfigError, match="top_k"):
        AttributionSelector(enabled=True, top_k=0)


def test_selector_rejects_mode_absent_from_study_modes() -> None:
    selector = AttributionSelector(enabled=True, trajectory_modes=("orientation",))
    with pytest.raises(StudyConfigError, match="absent from trajectory_modes"):
        _config(trajectory_modes=("magnitude",), attribution=selector)


def test_selector_rejects_effect_absent_from_study_effects() -> None:
    selector = AttributionSelector(enabled=True, effect_sizes=(0.75,))
    with pytest.raises(StudyConfigError, match="absent from effect_sizes"):
        _config(attribution=selector)


# --- PLS-only enforcement ---------------------------------------------------


@pytest.mark.parametrize("method", ["concat", "snf"])
def test_study_config_rejects_attribution_without_pls(method: str) -> None:
    evaluation = SimulationEvaluationParams(integration_method=method)  # type: ignore[arg-type]
    with pytest.raises(StudyConfigError, match="integration_method='pls'"):
        _config(evaluation=evaluation, attribution=AttributionSelector(enabled=True))


@pytest.mark.parametrize("method", ["concat", "snf"])
def test_evaluation_params_reject_attribution_without_pls(method: str) -> None:
    params = SimulationEvaluationParams(
        integration_method=method,  # type: ignore[arg-type]
        attribution=AttributionDiagnosticSettings(enabled=True),
    )
    with pytest.raises(SimulationEvaluationError, match="require integration_method='pls'"):
        _validate_evaluation_params(params)


# --- gate parameter validation ----------------------------------------------


def test_gate_requires_complete_parameters() -> None:
    payload = {"enabled": True, "alpha": 0.05}
    with pytest.raises(StudyConfigError, match="missing required parameter"):
        _load_from_payload({"acceptance": {"gate": payload}})


def test_gate_rejects_unknown_role_and_statistic() -> None:
    with pytest.raises(StudyConfigError, match="role"):
        GateRule(trajectory_mode="magnitude", statistic="delta", role="advisory")
    with pytest.raises(StudyConfigError, match="statistic"):
        GateRule(trajectory_mode="magnitude", statistic="size", role="descriptive")


def test_gate_requires_at_least_one_mandatory_power_rule() -> None:
    with pytest.raises(StudyConfigError, match="mandatory_power"):
        _gate(rules=(GateRule(trajectory_mode="magnitude", statistic="angle", role="descriptive"),))


def test_gate_rejects_duplicate_rules() -> None:
    rule = GateRule(trajectory_mode="magnitude", statistic="delta", role="mandatory_power")
    with pytest.raises(StudyConfigError, match="duplicate"):
        _gate(rules=(rule, rule))


def test_gate_power_floor_falls_back_to_the_global_minimum() -> None:
    gate = _gate()
    rule = gate.rules[0]
    assert gate.power_floor(rule) == pytest.approx(0.8)
    specific = GateRule(
        trajectory_mode="shape", statistic="shape", role="mandatory_power", min_power_at_top=0.5
    )
    assert gate.power_floor(specific) == pytest.approx(0.5)


# --- deterministic signatures -----------------------------------------------


def test_enumeration_and_signatures_are_deterministic() -> None:
    config = _config(attribution=AttributionSelector(enabled=True))
    first = enumerate_study(config)
    second = enumerate_study(config)
    assert [c.cell_id for c in first.cells] == [c.cell_id for c in second.cells]
    assert [parameter_signature(c) for c in first.cells] == [parameter_signature(c) for c in second.cells]


def test_attribution_settings_change_the_cell_signature() -> None:
    base = enumerate_study(_config(attribution=AttributionSelector(enabled=True, bootstrap_replicates=10)))
    changed = enumerate_study(_config(attribution=AttributionSelector(enabled=True, bootstrap_replicates=25)))

    def eligible(grid) -> dict[str, str]:
        return {
            f"{c.metadata['trajectory_mode']}@{c.metadata['effect_size']}": parameter_signature(c)
            for c in grid.cells
            if c.evaluation_params.attribution.enabled
        }

    base_signatures = eligible(base)
    changed_signatures = eligible(changed)
    assert set(base_signatures) == set(changed_signatures)
    assert all(base_signatures[key] != changed_signatures[key] for key in base_signatures)


def test_config_round_trips_through_json(tmp_path: Path) -> None:
    payload = {
        "generator": {"seed": 2, "trajectory_mode": "magnitude", "n_samples": 60, "n_stages": 4},
        "evaluation": {"integration_method": "pls", "permutations": 0, "seed": 3},
        "trajectory_modes": ["magnitude", "orientation", "shape", "translation"],
        "effect_sizes": [0.0, 0.5, 1.0],
        "n_replicates": 2,
        "base_seed": 100,
        "attribution": {"enabled": True, "bootstrap_replicates": 11, "top_k": 4},
        "matched_seeds": {"enabled": True, "version": 1},
        "acceptance": {
            "gate": {
                "enabled": True,
                "alpha": 0.05,
                "control_se_tolerance": 2.0,
                "monotonicity_se_tolerance": 2.0,
                "min_power_at_top": 0.8,
                "confirmation_se_threshold": 1.0,
                "rules": [
                    {"trajectory_mode": "magnitude", "statistic": "delta", "role": "mandatory_power"},
                    {"trajectory_mode": "magnitude", "statistic": "angle", "role": "mandatory_control"},
                    {"trajectory_mode": "orientation", "statistic": "delta", "role": "descriptive"},
                ],
            }
        },
    }
    path = tmp_path / "phase4.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    config = load_study_config(path)

    assert config.attribution == AttributionSelector(enabled=True, bootstrap_replicates=11, top_k=4)
    assert config.matched_seeds == MatchedSeedPolicy(enabled=True, version=1)
    gate = config.acceptance.gate
    assert gate.enabled is True
    assert {rule.role for rule in gate.rules} == {"mandatory_power", "mandatory_control", "descriptive"}


def _load_from_payload(extra: dict) -> StudyConfig:
    import tempfile

    payload = {
        "generator": {"seed": 2, "n_samples": 60},
        "evaluation": {"integration_method": "pls"},
        "trajectory_modes": ["magnitude"],
        "effect_sizes": [0.0, 1.0],
    }
    payload.update(extra)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(payload, handle)
        path = Path(handle.name)
    return load_study_config(path)
