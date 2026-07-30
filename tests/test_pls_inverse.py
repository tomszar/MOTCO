"""Tests for the controlled inverse-PLS interpretability study."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from motco.simulations.pls_inverse import (
    PLSInverseStudyError,
    PLSInverseStudyParams,
    additive_reconstruction,
    fit_inverse_pls,
    generate_paired_dataset,
    induced_metric,
    intervene_scores,
    results_frames,
    run_inverse_cell,
    run_inverse_study,
    stage_centroids,
    trajectory_geometry,
)

PARAMS = PLSInverseStudyParams(seed=7, n_stages=3, n_features=12, n_samples_per_stage=10)


def test_paired_groups_are_exactly_identical() -> None:
    dataset = generate_paired_dataset(PARAMS)
    a = dataset.X.loc[dataset.metadata["group"].eq("A").to_numpy()].to_numpy()
    b = dataset.X.loc[dataset.metadata["group"].eq("B").to_numpy()].to_numpy()
    np.testing.assert_array_equal(a, b)
    fit = fit_inverse_pls(dataset)
    np.testing.assert_allclose(
        stage_centroids(fit.scores, dataset.metadata, "A"),
        stage_centroids(fit.scores, dataset.metadata, "B"),
        atol=1e-12,
    )
    baseline = trajectory_geometry(fit.scores, dataset.metadata)
    assert baseline.delta < 1e-12
    assert baseline.angle < 1e-6
    assert baseline.shape < 1e-12


def test_generation_is_deterministic() -> None:
    first = generate_paired_dataset(PARAMS)
    second = generate_paired_dataset(PARAMS)
    np.testing.assert_array_equal(first.X.to_numpy(), second.X.to_numpy())


def test_magnitude_centroids_and_residuals() -> None:
    dataset = generate_paired_dataset(PARAMS)
    fit = fit_inverse_pls(dataset)
    original = stage_centroids(fit.scores, dataset.metadata, "B")
    modified, target = intervene_scores(fit.scores, dataset.metadata, "magnitude", PARAMS)
    center = original.mean(axis=0)
    np.testing.assert_allclose(target, center + PARAMS.magnitude_scale * (original - center))
    np.testing.assert_allclose(target.mean(axis=0), center, atol=1e-12)
    _assert_score_residuals_preserved(fit.scores, modified, dataset.metadata)


def test_orientation_is_rigid_and_preserves_residuals() -> None:
    dataset = generate_paired_dataset(PARAMS)
    fit = fit_inverse_pls(dataset)
    original = stage_centroids(fit.scores, dataset.metadata, "B")
    modified, target = intervene_scores(fit.scores, dataset.metadata, "orientation", PARAMS)
    np.testing.assert_allclose(_pairwise_distances(target), _pairwise_distances(original), atol=1e-10)
    np.testing.assert_allclose(target.mean(axis=0), original.mean(axis=0), atol=1e-12)
    _assert_score_residuals_preserved(fit.scores, modified, dataset.metadata)


def test_shape_holds_endpoints_and_moves_middle_perpendicularly() -> None:
    dataset = generate_paired_dataset(PARAMS)
    fit = fit_inverse_pls(dataset)
    original = stage_centroids(fit.scores, dataset.metadata, "B")
    modified, target = intervene_scores(fit.scores, dataset.metadata, "shape", PARAMS)
    np.testing.assert_allclose(target[[0, 2]], original[[0, 2]], atol=1e-12)
    displacement = target[1] - original[1]
    endpoint = original[2] - original[0]
    assert abs(float(displacement @ endpoint)) < 1e-10
    _assert_score_residuals_preserved(fit.scores, modified, dataset.metadata)


def test_shape_rejects_two_stages() -> None:
    params = replace(PARAMS, n_stages=2)
    dataset = generate_paired_dataset(params)
    fit = fit_inverse_pls(dataset)
    with pytest.raises(PLSInverseStudyError, match="three stages"):
        intervene_scores(fit.scores, dataset.metadata, "shape", params)


def test_additive_reconstruction_roundtrip_and_residual() -> None:
    dataset = generate_paired_dataset(PARAMS)
    fit = fit_inverse_pls(dataset)
    modified, _ = intervene_scores(fit.scores, dataset.metadata, "orientation", PARAMS)
    X_star, roundtrip, residual = additive_reconstruction(
        dataset.X.to_numpy(), fit.scores, modified, dataset.metadata, fit.model
    )
    a_mask = dataset.metadata["group"].eq("A").to_numpy()
    np.testing.assert_array_equal(X_star[a_mask], dataset.X.to_numpy()[a_mask])
    assert roundtrip < 1e-10
    assert residual < 1e-12


def test_geometry_availability_by_stage_count() -> None:
    two = run_inverse_cell(replace(PARAMS, n_stages=2), "magnitude")
    three = run_inverse_cell(PARAMS, "shape")
    assert np.isnan(two.latent_geometry.shape)
    assert np.isnan(two.feature_geometry.shape)
    assert np.isfinite(three.latent_geometry.shape)
    assert np.isfinite(three.feature_geometry.shape)


def test_feature_table_and_study_output_schema() -> None:
    results = run_inverse_study(PARAMS)
    cells, features = results_frames(results)
    assert len(results) == 5
    assert set(zip(cells["n_stages"], cells["intervention"])) == {
        (2, "magnitude"),
        (2, "orientation"),
        (3, "magnitude"),
        (3, "orientation"),
        (3, "shape"),
    }
    assert {"feature", "mean_absolute_change", "loading_lv1", "loading_lv2"} <= set(features.columns)
    assert len(features) == 5 * PARAMS.n_features
    assert cells["roundtrip_max_error"].max() < 1e-10


def test_induced_metric_isotropic_and_anisotropic() -> None:
    metric, eigenvalues, condition = induced_metric(np.eye(2))
    np.testing.assert_allclose(metric, np.eye(2))
    np.testing.assert_allclose(eigenvalues, [1.0, 1.0])
    assert condition == pytest.approx(1.0)
    metric, eigenvalues, condition = induced_metric(np.diag([1.0, 3.0]))
    np.testing.assert_allclose(metric, np.diag([1.0, 9.0]))
    np.testing.assert_allclose(eigenvalues, [1.0, 9.0])
    assert condition == pytest.approx(9.0)


def _pairwise_distances(points: np.ndarray) -> np.ndarray:
    return np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)


def _assert_score_residuals_preserved(original: np.ndarray, modified: np.ndarray, metadata) -> None:
    groups = metadata["group"].to_numpy()
    stages = metadata["stage"].to_numpy()
    for stage in sorted(set(stages)):
        mask = (groups == "B") & (stages == stage)
        original_residual = original[mask] - original[mask].mean(axis=0)
        modified_residual = modified[mask] - modified[mask].mean(axis=0)
        np.testing.assert_allclose(modified_residual, original_residual, atol=1e-12)
