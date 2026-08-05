from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from motco.simulations.diagnostics import geometry_from_means
from motco.simulations.generator import logit
from motco.simulations.preprocessing import concatenate_blocks, fit_omics_preprocessor
from motco.simulations.realized_geometry_study import (
    monotonicity_summary,
    summarize_phase2_characterization,
)
from motco.simulations.reference import load_reference
from motco.simulations.semisynthetic import (
    SemiSyntheticTrajectoryDataset,
    SemiSyntheticTrajectoryParams,
    generate_semisynthetic_trajectory,
)


@pytest.fixture(scope="module")
def reference():
    return load_reference()


def _means(a: list[list[float]], b: list[list[float]]) -> pd.DataFrame:
    n_stages = len(a)
    return pd.DataFrame(
        np.asarray(a + b, dtype=float),
        index=pd.MultiIndex.from_product(
            [["A", "B"], [str(stage) for stage in range(n_stages)]],
            names=["group", "stage"],
        ),
    )


@pytest.mark.parametrize(
    ("means", "expected_delta", "expected_angle", "shape_positive"),
    [
        (_means([[0, 0], [1, 0], [2, 0]], [[0, 0], [2, 0], [4, 0]]), 2.0, 0.0, False),
        (_means([[0, 0], [1, 0], [2, 0]], [[0, 0], [0, 1], [0, 2]]), 0.0, 90.0, False),
        (_means([[0, 0], [1, 0], [2, 0]], [[0, 0], [1, 1], [2, 0]]), 2**0.5 * 2 - 2, 0.0, True),
        (_means([[0, 0], [1, 0], [2, 0]], [[5, 2], [6, 2], [7, 2]]), 0.0, 0.0, False),
    ],
)
def test_geometry_from_known_trajectories(
    means: pd.DataFrame,
    expected_delta: float,
    expected_angle: float,
    shape_positive: bool,
) -> None:
    result = geometry_from_means(means, ["A", "B"], ["0", "1", "2"])

    assert result.delta == pytest.approx(expected_delta)
    assert result.angle == pytest.approx(expected_angle)
    if shape_positive:
        assert (result.shape or 0.0) > 0.0
    else:
        assert result.shape == pytest.approx(0.0)


def test_geometry_marks_undefined_angle_and_two_stage_shape() -> None:
    zero = geometry_from_means(
        _means([[0, 0], [0, 0]], [[1, 1], [1, 1]]),
        ["A", "B"],
        ["0", "1"],
    )

    assert zero.angle is None
    assert zero.shape is None
    assert zero.availability == {"delta": True, "angle": False, "shape": False}


def test_population_trajectories_match_generator_mean_contract(reference) -> None:
    params = SemiSyntheticTrajectoryParams(
        seed=7,
        trajectory_mode="orientation",
        n_samples=240,
        n_stages=3,
        group_effect_size=0.6,
    )
    dataset = generate_semisynthetic_trajectory(params, reference=reference)
    assert dataset.population_trajectories is not None
    population = dataset.population_trajectories.layers
    indicators = dataset.truth["indicators"]
    deltas = dataset.truth["deltas"]

    expected_a0 = reference.mean_M + deltas["A"][0] * indicators["A"]["methylation"][:, 0]
    np.testing.assert_allclose(population["methylation"].loc[("A", "0")], expected_a0)
    assert list(population["methylation"].columns) == list(reference.cpg_names)
    assert list(population["expression"].columns) == list(reference.gene_names)
    assert list(population["proteomics"].columns) == list(reference.protein_names)

    observed = {
        "methylation": pd.DataFrame(
            logit(dataset.methylation.to_numpy()), index=dataset.metadata.index
        ),
        "expression": dataset.expression,
        "proteomics": dataset.proteomics,
    }
    for layer, matrix in observed.items():
        grouped = matrix.groupby([dataset.metadata["group"], dataset.metadata["stage"]]).mean()
        grouped.index = pd.MultiIndex.from_tuples(
            [(str(group), str(stage)) for group, stage in grouped.index], names=["group", "stage"]
        )
        rmse = np.sqrt(np.mean((grouped.to_numpy() - population[layer].to_numpy()) ** 2))
        assert rmse < 0.35


def test_shared_preprocessor_transforms_population_and_observations_identically() -> None:
    ids = ["a0", "a1", "b0", "b1"]
    metadata = pd.DataFrame(
        {"sample_id": ids, "group": ["A", "A", "B", "B"], "stage": [0, 1, 0, 1]},
        index=ids,
    )
    constant = pd.DataFrame({"f": [0.5] * 4}, index=ids)
    dataset = SemiSyntheticTrajectoryDataset(
        methylation=constant,
        expression=pd.DataFrame({"g": [2.0] * 4}, index=ids),
        proteomics=pd.DataFrame({"p": [3.0] * 4}, index=ids),
        metadata=metadata,
    )
    preprocessor = fit_omics_preprocessor(dataset)
    observed = preprocessor.transform_dataset(dataset)
    joint = concatenate_blocks(observed)

    assert np.isfinite(joint.to_numpy()).all()
    np.testing.assert_allclose(joint.to_numpy(), 0.0)
    assert all(scaler.scale[0] == 1.0 for scaler in preprocessor.scalers.values())


def test_generated_evaluation_contains_full_preintegration_decomposition(reference) -> None:
    from motco.simulations.evaluation import SimulationEvaluationParams, evaluate_semisynthetic_trajectory

    dataset = generate_semisynthetic_trajectory(
        SemiSyntheticTrajectoryParams(
            seed=9,
            trajectory_mode="shape",
            n_samples=120,
            n_stages=3,
            group_effect_size=0.5,
        ),
        reference=reference,
    )
    result = evaluate_semisynthetic_trajectory(
        dataset,
        SimulationEvaluationParams(integration_method="concat", permutations=0),
    )

    diagnostics = result.realized_geometry
    assert diagnostics["schema_version"] == 1
    assert set(diagnostics["checkpoints"]) == {
        "population_native",
        "population_standardized",
        "observed_standardized",
    }
    assert set(diagnostics["checkpoints"]["population_native"]) == {
        "methylation",
        "expression",
        "proteomics",
    }
    assert "joint" in diagnostics["checkpoints"]["population_standardized"]
    assert diagnostics["requested"]["trajectory_mode"] == "shape"


def test_phase2_summary_flattens_geometry_and_excludes_unavailable_values() -> None:
    rows = [
        {
            "seed": seed,
            "mode": "orientation",
            "shape_kind": None,
            "effect_size": effect,
            "population_native.methylation.angle": angle,
            "population_native.methylation.shape": None,
            "population_native.methylation.path_length.A": 2.0,
        }
        for seed, effect, angle in [(0, 0.0, 0.0), (0, 0.5, 20.0), (0, 1.0, 40.0)]
    ]

    summary = summarize_phase2_characterization(rows)
    angle = summary[summary["statistic"] == "angle"]
    shape = summary[summary["statistic"] == "shape"]
    monotonicity = monotonicity_summary(summary)

    assert angle["n_available"].tolist() == [1, 1, 1]
    assert shape["n_available"].tolist() == [0, 0, 0]
    angle_monotonicity = monotonicity[monotonicity["statistic"] == "angle"]
    assert angle_monotonicity.iloc[0]["spearman"] == pytest.approx(1.0)
