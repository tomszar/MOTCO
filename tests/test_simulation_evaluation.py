from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from motco.simulations import (
    SemiSyntheticTrajectoryDataset,
    SimulationEvaluationError,
    SimulationEvaluationParams,
    build_simulation_trajectory_design,
    evaluate_semisynthetic_trajectory,
    integrate_semisynthetic_dataset,
)


def make_dataset(*, n_stages: int = 3) -> SemiSyntheticTrajectoryDataset:
    rows = []
    for stage in range(n_stages):
        for group in ("A", "B"):
            for replicate in range(2):
                rows.append((f"s{stage}_{group}_{replicate}", group, stage, replicate))
    sample_ids = [row[0] for row in rows]
    metadata = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "group": [row[1] for row in rows],
            "stage": [row[2] for row in rows],
            "cluster": [row[2] for row in rows],
        },
        index=sample_ids,
    )

    values = []
    for _, group, stage, replicate in rows:
        group_offset = 1.0 if group == "B" else 0.0
        values.append(
            [
                float(stage + group_offset + replicate * 0.1),
                float((stage + 1) * (group_offset + 1.0)),
                float(replicate - stage * 0.2),
            ]
        )
    base = pd.DataFrame(values, index=sample_ids, columns=["f0", "f1", "f2"])
    return SemiSyntheticTrajectoryDataset(
        methylation=base.add_prefix("m"),
        expression=(base + 0.5).add_prefix("g"),
        proteomics=(base * 0.5).add_prefix("p"),
        metadata=metadata,
        truth={"trajectory_mode": "magnitude", "seed": 17},
    )


def test_concat_integration_standardizes_and_records_metadata() -> None:
    result = integrate_semisynthetic_dataset(make_dataset(), SimulationEvaluationParams(integration_method="concat"))

    assert result.matrix.shape == (12, 9)
    assert result.metadata["integration_method"] == "concat"
    assert result.metadata["integration_params"] == {"standardize": True}
    np.testing.assert_allclose(result.matrix.mean(axis=0).to_numpy(), 0.0, atol=1e-12)


def test_snf_integration_uses_existing_helpers_and_records_resolved_params() -> None:
    result = integrate_semisynthetic_dataset(
        make_dataset(),
        SimulationEvaluationParams(
            integration_method="snf",
            integration_params={"K": 3, "k": 3, "t": 2, "spectral_components": 2},
        ),
    )

    assert result.matrix.shape == (12, 2)
    assert result.metadata["integration_method"] == "snf"
    assert result.metadata["integration_params"]["spectral_components"] == 2
    assert result.metadata["fused_shape"] == (12, 12)


def test_unsupported_integration_method_is_rejected() -> None:
    params = SimulationEvaluationParams(integration_method="bad")  # type: ignore[arg-type]

    with pytest.raises(SimulationEvaluationError, match="Unsupported integration_method"):
        evaluate_semisynthetic_trajectory(make_dataset(), params)


def test_invalid_snf_parameters_are_rejected() -> None:
    params = SimulationEvaluationParams(integration_method="snf", integration_params={"K": 12})

    with pytest.raises(SimulationEvaluationError, match="'K' must be at most 11"):
        integrate_semisynthetic_dataset(make_dataset(), params)


def test_missing_metadata_columns_are_rejected() -> None:
    dataset = make_dataset()
    bad = SemiSyntheticTrajectoryDataset(
        methylation=dataset.methylation,
        expression=dataset.expression,
        proteomics=dataset.proteomics,
        metadata=dataset.metadata.drop(columns=["group"]),
        truth=dataset.truth,
    )

    with pytest.raises(SimulationEvaluationError, match="missing required column"):
        evaluate_semisynthetic_trajectory(bad)


def test_omics_metadata_row_mismatch_is_rejected() -> None:
    dataset = make_dataset()
    bad = SemiSyntheticTrajectoryDataset(
        methylation=dataset.methylation.iloc[::-1],
        expression=dataset.expression,
        proteomics=dataset.proteomics,
        metadata=dataset.metadata,
        truth=dataset.truth,
    )

    with pytest.raises(SimulationEvaluationError, match="not aligned"):
        evaluate_semisynthetic_trajectory(bad)


def test_design_objects_and_contrast_are_built_from_sorted_levels() -> None:
    design = build_simulation_trajectory_design(make_dataset().metadata)

    assert design.group_levels == ["A", "B"]
    assert design.stage_levels == ["0", "1", "2"]
    assert design.model_full.shape == (12, 6)
    assert design.model_reduced.shape == (12, 4)
    assert design.ls_means.shape == (6, 6)
    assert design.contrast == [[0, 1, 2], [3, 4, 5]]


def test_observed_only_evaluation_returns_statistics_without_p_values() -> None:
    result = evaluate_semisynthetic_trajectory(
        make_dataset(),
        SimulationEvaluationParams(integration_method="concat", permutations=0),
    )

    assert result.observed_deltas.shape == (2, 2)
    assert result.observed_angles.shape == (2, 2)
    assert result.observed_shapes.shape == (2, 2)
    assert set(result.pair_statistics) == {"delta", "angle", "shape"}
    assert result.p_values == {}
    assert result.truth_metadata["trajectory_mode"] == "magnitude"
    assert result.runtime_metadata["permutations"] == 0


def test_rrpp_p_values_use_plus_one_correction_and_seed() -> None:
    params = SimulationEvaluationParams(
        integration_method="concat",
        permutations=3,
        seed=123,
        include_null_distributions=True,
    )
    first = evaluate_semisynthetic_trajectory(make_dataset(), params)
    second = evaluate_semisynthetic_trajectory(make_dataset(), params)

    assert first.p_values == second.p_values
    assert first.null_distributions == second.null_distributions
    assert set(first.p_values) == {"delta", "angle", "shape"}
    for statistic, observed in first.pair_statistics.items():
        assert first.null_distributions is not None
        null_values = np.asarray(first.null_distributions[statistic], dtype=float)
        expected = float((1.0 + np.sum(null_values >= observed)) / 4.0)
        assert first.p_values[statistic] == expected
        assert 0.0 < first.p_values[statistic] <= 1.0


def test_shape_pair_statistic_is_unavailable_for_two_stages() -> None:
    result = evaluate_semisynthetic_trajectory(
        make_dataset(n_stages=2),
        SimulationEvaluationParams(integration_method="concat", permutations=2, seed=321),
    )

    assert np.isnan(result.pair_statistics["shape"])
    assert "shape" not in result.p_values
    assert not result.runtime_metadata["shape_available"]


def test_truth_metadata_is_json_serializable_without_indicator_arrays() -> None:
    """The persisted truth_metadata must be JSON-safe (no raw numpy indicator arrays)."""

    import json

    dataset = make_dataset()
    # mimic the real generator's truth: JSON-able summary plus raw ndarray indicators
    dataset = SemiSyntheticTrajectoryDataset(
        methylation=dataset.methylation,
        expression=dataset.expression,
        proteomics=dataset.proteomics,
        metadata=dataset.metadata,
        truth={
            "trajectory_mode": "orientation",
            "indicator_counts": {"A": {"methylation": [1, 2, 1]}},
            "indicators": {"A": {"methylation": np.zeros((4, 3))}},
        },
    )
    result = evaluate_semisynthetic_trajectory(dataset, SimulationEvaluationParams(permutations=0))
    assert "indicators" not in result.truth_metadata
    assert result.truth_metadata["indicator_counts"] == {"A": {"methylation": [1, 2, 1]}}
    # round-trips through JSON (this is what the study's JSONL persistence does)
    json.dumps(result.truth_metadata)
