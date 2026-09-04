"""Single-fit PLS boundary and the orientation-attribution adapter."""

from __future__ import annotations

import numpy as np
import pytest

from motco.simulations import (
    AttributionDiagnosticSettings,
    SemiSyntheticTrajectoryParams,
    SimulationEvaluationParams,
)
from motco.simulations.attribution_diagnostics import (
    ATTRIBUTION_SCHEMA_VERSION,
    derive_truth_driver_features,
)
from motco.simulations.evaluation import (
    SimulationEvaluationError,
    evaluate_semisynthetic_trajectory,
    integrate_semisynthetic_dataset,
)
from motco.simulations.semisynthetic import generate_semisynthetic_trajectory

pytestmark = pytest.mark.slow

# The orientation surgery is pool-limited: at p_dmp=0.2 over three stages the
# destination pool saturates just under e = 0.89 (``expected_surgery_headroom``),
# so a full-size effect would trip ``surgery_censoring="error"``. 0.8 keeps a
# strong, fully realized orientation effect inside the guard band.
_GENERATOR = SemiSyntheticTrajectoryParams(
    seed=7,
    trajectory_mode="orientation",
    n_samples=60,
    n_stages=3,
    group_effect_size=0.8,
    p_dmp=0.2,
)

_PLS_PARAMS: dict[str, object] = {
    "integration_method": "pls",
    "permutations": 0,
    "seed": 11,
    "integration_params": {"cv1_splits": 2, "cv2_splits": 2, "n_repeats": 1, "max_components": 3},
}


@pytest.fixture(scope="module")
def dataset():
    return generate_semisynthetic_trajectory(_GENERATOR)


@pytest.fixture(scope="module")
def disabled_result(dataset):
    return evaluate_semisynthetic_trajectory(dataset, SimulationEvaluationParams(**_PLS_PARAMS))


@pytest.fixture(scope="module")
def enabled_result(dataset):
    params = SimulationEvaluationParams(
        **_PLS_PARAMS,
        attribution=AttributionDiagnosticSettings(enabled=True, bootstrap_replicates=3, top_k=5),
    )
    return evaluate_semisynthetic_trajectory(dataset, params)


# --- 2.3 attribution-disabled equivalence ------------------------------------


def test_disabled_and_enabled_evaluations_agree(disabled_result, enabled_result) -> None:
    """Enabling attribution must not perturb any production measurement."""

    assert (
        disabled_result.latent_matrix_metadata["selected_lv"]
        == enabled_result.latent_matrix_metadata["selected_lv"]
    )
    assert disabled_result.pair_statistics.keys() == enabled_result.pair_statistics.keys()
    for statistic, value in disabled_result.pair_statistics.items():
        other = enabled_result.pair_statistics[statistic]
        if np.isnan(value):
            assert np.isnan(other)
        else:
            assert other == pytest.approx(value, rel=1e-12, abs=1e-12)
    np.testing.assert_allclose(disabled_result.observed_deltas, enabled_result.observed_deltas)
    np.testing.assert_allclose(disabled_result.observed_angles, enabled_result.observed_angles)


def test_repeated_evaluation_with_the_same_seed_is_reproducible(dataset) -> None:
    params = SimulationEvaluationParams(
        **_PLS_PARAMS,
        attribution=AttributionDiagnosticSettings(enabled=True, bootstrap_replicates=3, top_k=4),
    )
    first = evaluate_semisynthetic_trajectory(dataset, params).attribution_diagnostics
    second = evaluate_semisynthetic_trajectory(dataset, params).attribution_diagnostics
    assert first["top_features"] == second["top_features"]
    assert first["stability"] == second["stability"]
    assert first["truth_recovery"] == second["truth_recovery"]


def test_p_values_match_with_attribution_enabled(dataset) -> None:
    common = {**_PLS_PARAMS, "permutations": 19}
    without = evaluate_semisynthetic_trajectory(dataset, SimulationEvaluationParams(**common))
    with_attr = evaluate_semisynthetic_trajectory(
        dataset,
        SimulationEvaluationParams(
            **common,
            attribution=AttributionDiagnosticSettings(enabled=True, bootstrap_replicates=2, top_k=3),
        ),
    )
    assert without.p_values == with_attr.p_values


# --- 2.4 shared fit and non-PLS rejection ------------------------------------


def test_attribution_receives_the_estimator_used_for_measurement(dataset) -> None:
    params = SimulationEvaluationParams(**_PLS_PARAMS)
    latent = integrate_semisynthetic_dataset(dataset, params)
    artifacts = latent.artifacts
    assert artifacts is not None
    np.testing.assert_allclose(np.asarray(artifacts.model.x_scores_), latent.matrix.to_numpy())
    assert artifacts.features.shape[1] == int(artifacts.model.n_features_in_)
    assert artifacts.features.shape[1] == artifacts.original_scales.shape[0]
    assert artifacts.methylation_units == "mvalue"


def test_evaluation_result_is_json_safe_and_holds_no_estimator(enabled_result) -> None:
    import json

    from motco.simulations.grid import _to_jsonable

    assert not hasattr(enabled_result, "model")
    payload = {
        "latent_matrix_metadata": enabled_result.latent_matrix_metadata,
        "attribution_diagnostics": enabled_result.attribution_diagnostics,
    }
    json.dumps(_to_jsonable(payload), allow_nan=False)


@pytest.mark.parametrize("method", ["concat", "snf"])
def test_attribution_rejected_for_non_pls_methods(dataset, method: str) -> None:
    params = SimulationEvaluationParams(
        integration_method=method,  # type: ignore[arg-type]
        permutations=0,
        seed=11,
        attribution=AttributionDiagnosticSettings(enabled=True, bootstrap_replicates=2),
    )
    with pytest.raises(SimulationEvaluationError, match="require integration_method='pls'"):
        evaluate_semisynthetic_trajectory(dataset, params)


# --- 3.x diagnostic content ---------------------------------------------------


def test_attribution_record_is_bounded_and_versioned(enabled_result) -> None:
    record = enabled_result.attribution_diagnostics
    assert record["schema_version"] == ATTRIBUTION_SCHEMA_VERSION
    assert record["status"] == "computed"
    assert record["settings"]["top_k"] == 5
    assert record["model"]["selected_components"] == enabled_result.latent_matrix_metadata["selected_lv"]
    assert record["model"]["feature_order_signature"]
    assert record["units"]["methylation_units"] == "mvalue"
    assert record["units"]["original_basis"]["methylation"] == "mvalue"

    # n_stages=3 → 2 transitions, each with three components.
    assert len(record["transitions"]) == 2
    assert set(record["transitions"][0]["components"]) == {"observed", "pls_captured", "residual"}

    by_key: dict[tuple[str, str], int] = {}
    for entry in record["top_features"]:
        by_key[(entry["transition_id"], entry["component"])] = (
            by_key.get((entry["transition_id"], entry["component"]), 0) + 1
        )
    assert by_key, "expected top-k feature records"
    assert max(by_key.values()) <= 5, "top-k truncation must bound the payload"
    assert record["model"]["n_features"] > 5


def test_top_feature_records_carry_both_unit_bases(enabled_result) -> None:
    entries = [e for e in enabled_result.attribution_diagnostics["top_features"] if e["component"] == "observed"]
    assert entries
    assert all(e["effect_standardized"] is not None for e in entries)
    assert any(e["effect_original"] is not None for e in entries)
    assert all(e["sign"] in (-1, 0, 1) for e in entries)


def test_disabled_evaluation_records_not_requested(disabled_result) -> None:
    record = disabled_result.attribution_diagnostics
    assert record["status"] == "not_requested"
    assert "top_features" not in record


def test_truth_recovery_uses_propagated_drivers(dataset, enabled_result) -> None:
    record = enabled_result.attribution_diagnostics
    assert record["truth"]["available"] is True
    assert record["truth"]["n_drivers"] > 0
    assert all(entry["truth_count"] >= 0 for entry in record["truth_recovery"])

    feature_names = [
        f"{layer}__{name}"
        for layer in ("methylation", "expression", "proteomics")
        for name in getattr(dataset, layer).columns.astype(str)
    ]
    truth = derive_truth_driver_features(dataset, feature_names, ["0", "1", "2"])
    layers = {name.split("__", 1)[0] for name in truth.names()}
    # Orientation relocates methylation sites; the cascade must carry the change
    # into expression and proteomics so propagated drivers are truth, not noise.
    assert "methylation" in layers
    assert layers & {"expression", "proteomics"}
