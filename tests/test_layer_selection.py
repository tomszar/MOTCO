"""Evaluation-time omic block selection via ``integration_params["layers"]``."""

from __future__ import annotations

import pytest

from motco.simulations import (
    SemiSyntheticTrajectoryParams,
    SimulationEvaluationError,
    SimulationEvaluationParams,
    evaluate_semisynthetic_trajectory,
    generate_semisynthetic_trajectory,
    integrate_semisynthetic_dataset,
    load_reference,
)
from motco.simulations.evaluation import AttributionDiagnosticSettings
from motco.simulations.preprocessing import (
    OMIC_LAYERS,
    concatenate_blocks,
    fit_omics_preprocessor,
    selected_layers,
)

TWO = ["expression", "methylation"]
CANONICAL_TWO = ("methylation", "expression")


@pytest.fixture(scope="module")
def dataset():
    params = SemiSyntheticTrajectoryParams(
        seed=4,
        trajectory_mode="orientation",
        group_effect_size=0.3,
        p_dmp=0.1,
        group_stage_sizes=((11, 10, 28), (9, 10, 12)),
    )
    return generate_semisynthetic_trajectory(params, reference=load_reference())


def _prefixes(columns) -> set[str]:
    return {str(column).split("__", 1)[0] for column in columns}


# ── Resolver ─────────────────────────────────────────────────────────────────


def test_absent_selection_is_every_layer() -> None:
    assert selected_layers({}) == OMIC_LAYERS
    assert selected_layers({"layers": None}) == OMIC_LAYERS


def test_selection_is_normalized_to_canonical_order() -> None:
    assert selected_layers({"layers": TWO}) == CANONICAL_TWO


@pytest.mark.parametrize(
    ("layers", "match"),
    [([], "must not be empty"), (["methylation", "atac"], "unknown"), (["methylation", "methylation"], "repeats")],
    ids=["empty", "unknown", "duplicate"],
)
def test_invalid_selection_is_rejected_naming_the_allowed_layers(dataset, layers, match) -> None:
    with pytest.raises(ValueError, match=match) as info:
        selected_layers({"layers": layers})
    assert "proteomics" in str(info.value)
    params = SimulationEvaluationParams(integration_params={"layers": layers})
    with pytest.raises(SimulationEvaluationError, match=match):
        evaluate_semisynthetic_trajectory(dataset, params)


# ── Preprocessing ────────────────────────────────────────────────────────────


def test_preprocessor_fits_and_transforms_only_the_selected_layers(dataset) -> None:
    preprocessor = fit_omics_preprocessor(dataset, CANONICAL_TWO)
    assert preprocessor.layers == CANONICAL_TWO
    blocks = preprocessor.transform_dataset(dataset)
    assert tuple(blocks) == CANONICAL_TWO
    assert tuple(preprocessor.transform_population(dataset.population_trajectories)) == CANONICAL_TWO
    joint = concatenate_blocks(blocks)
    assert joint.shape[1] == dataset.methylation.shape[1] + dataset.expression.shape[1]
    assert _prefixes(joint.columns) == set(CANONICAL_TWO)


def test_a_preprocessor_fitted_on_other_layers_is_rejected(dataset) -> None:
    preprocessor = fit_omics_preprocessor(dataset)
    params = SimulationEvaluationParams(integration_params={"layers": TWO})
    with pytest.raises(SimulationEvaluationError, match="fitted on layers"):
        integrate_semisynthetic_dataset(dataset, params, preprocessor=preprocessor)


# ── Integration methods ──────────────────────────────────────────────────────


@pytest.mark.parametrize("standardize", [True, False])
def test_concat_contains_only_selected_features(dataset, standardize) -> None:
    params = SimulationEvaluationParams(integration_params={"layers": TWO, "standardize": standardize})
    latent = integrate_semisynthetic_dataset(dataset, params)
    assert _prefixes(latent.matrix.columns) == set(CANONICAL_TWO)
    assert latent.metadata["layer_feature_counts"] == {"methylation": 367, "expression": 131}
    assert latent.metadata["integration_params"]["layers"] == list(CANONICAL_TWO)


def test_snf_fuses_only_the_selected_layers(dataset, monkeypatch) -> None:
    import motco.simulations.evaluation as evaluation

    seen: list[int] = []
    original = evaluation.get_affinity_matrix

    def spy(layers, **kwargs):
        seen.extend(layer.shape[1] for layer in layers)
        return original(layers, **kwargs)

    monkeypatch.setattr(evaluation, "get_affinity_matrix", spy)
    params = SimulationEvaluationParams(integration_method="snf", integration_params={"layers": TWO})
    latent = integrate_semisynthetic_dataset(dataset, params)
    assert seen == [367, 131]
    assert latent.metadata["integration_params"]["layers"] == list(CANONICAL_TWO)


def test_absent_selection_records_no_layers_key(dataset) -> None:
    latent = integrate_semisynthetic_dataset(dataset, SimulationEvaluationParams())
    assert "layers" not in latent.metadata["integration_params"]


def test_pls_evaluation_measures_only_the_selected_layers(dataset) -> None:
    params = SimulationEvaluationParams(
        integration_method="pls",
        integration_params={"layers": TWO, "n_repeats": 1},
        attribution=AttributionDiagnosticSettings(enabled=True, top_k=10),
    )
    result = evaluate_semisynthetic_trajectory(dataset, params)

    metadata = result.latent_matrix_metadata
    assert metadata["layer_feature_counts"] == {"methylation": 367, "expression": 131}
    assert metadata["integration_params"]["layers"] == list(CANONICAL_TWO)

    # Realized geometry: per-block scopes for the selected layers only.
    for checkpoint, scopes in result.realized_geometry["checkpoints"].items():
        assert "proteomics" not in scopes, checkpoint
    assert set(result.realized_geometry["checkpoints"]["observed_standardized"]) == {*CANONICAL_TWO, "joint"}
    assert set(result.realized_geometry["checkpoints"]["population_native"]) == set(CANONICAL_TWO)

    # Attribution: every feature record is methylation or expression.
    attribution = result.attribution_diagnostics
    assert attribution["status"] == "computed"
    assert attribution["model"]["n_features"] == 367 + 131
    assert attribution["top_features"]
    assert _prefixes(record["feature"] for record in attribution["top_features"]) <= set(CANONICAL_TWO)


def test_two_layer_joint_scope_differs_from_three_layer_joint_scope(dataset) -> None:
    two = evaluate_semisynthetic_trajectory(dataset, SimulationEvaluationParams(integration_params={"layers": TWO}))
    three = evaluate_semisynthetic_trajectory(dataset, SimulationEvaluationParams())
    observed_two = two.realized_geometry["checkpoints"]["observed_standardized"]
    observed_three = three.realized_geometry["checkpoints"]["observed_standardized"]
    # Per-block scopes are unchanged by dropping a block; the joint scope is not.
    assert observed_two["methylation"] == observed_three["methylation"]
    assert observed_two["joint"] != observed_three["joint"]
    assert two.pair_statistics != three.pair_statistics
