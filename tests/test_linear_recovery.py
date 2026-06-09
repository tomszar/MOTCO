"""Tests for the Rung-0 Gaussian linear-recovery test bed."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from motco.simulations.linear_recovery import (
    LinearRecoveryError,
    LinearRecoveryParams,
    delta_x_summary,
    generate_dataset,
    givens_rotation,
    inverse_design_magnitude,
    inverse_design_orientation,
    project_and_measure,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Full PCA (k = p) eliminates subspace leakage, making the clean-floor test
# analytically exact (up to finite-sample LS-mean noise).
_CLEAN_PARAMS = LinearRecoveryParams(
    seed=7,
    n_features=10,
    n_samples_per_cell=50,
    noise_scale=0.5,
    signal_scale=10.0,
    n_components=10,
    scale_c=2.0,
    angle_theta=45.0,
)


# ---------------------------------------------------------------------------
# 5.1  Deterministic output and structural sanity
# ---------------------------------------------------------------------------


def test_deterministic_same_seed():
    p1 = _CLEAN_PARAMS
    p2 = replace(p1)
    d1 = generate_dataset(p1)
    d2 = generate_dataset(p2)
    np.testing.assert_array_equal(d1.X.to_numpy(), d2.X.to_numpy())


def test_different_seeds_differ():
    d1 = generate_dataset(_CLEAN_PARAMS)
    d2 = generate_dataset(replace(_CLEAN_PARAMS, seed=99))
    assert not np.allclose(d1.X.to_numpy(), d2.X.to_numpy())


def test_two_stages_per_group():
    d = generate_dataset(_CLEAN_PARAMS)
    for group in ("A", "B"):
        stages = d.metadata.loc[d.metadata["group"] == group, "stage"].unique()
        assert set(stages) == {"0", "1"}


def test_feature_matrix_shape():
    p = _CLEAN_PARAMS
    d = generate_dataset(p)
    n_total = 4 * p.n_samples_per_cell
    assert d.X.shape == (n_total, p.n_features)


def test_per_cell_means_approximate_spec():
    # With n=50 samples per cell and noise_scale=0.5, per-cell mean error is
    # ~0.5/sqrt(50) ≈ 0.07 per feature; 3-sigma bound across 10 features ~ 0.22.
    p = _CLEAN_PARAMS
    d = generate_dataset(p)
    tol = 0.5  # generous: signal_scale=10 so 0.5 is < 5%
    for group in ("A", "B"):
        for stage in ("0", "1"):
            mask = (d.metadata["group"] == group) & (d.metadata["stage"] == stage)
            cell_mean = d.X.loc[mask.values].mean(axis=0).to_numpy()
            if stage == "0":
                expected = np.zeros(p.n_features)
            else:
                expected = d.step_A if group == "A" else d.step_B
            assert np.allclose(cell_mean, expected, atol=tol), (
                f"Cell ({group},{stage}) mean deviates by more than {tol}"
            )


def test_step_A_magnitude():
    d = generate_dataset(_CLEAN_PARAMS)
    assert np.isclose(np.linalg.norm(d.step_A), _CLEAN_PARAMS.signal_scale, rtol=1e-10)


def test_step_none_equals_step_A():
    d = generate_dataset(replace(_CLEAN_PARAMS, manipulation="none"))
    np.testing.assert_array_equal(d.step_A, d.step_B)


def test_step_magnitude_direction():
    p = replace(_CLEAN_PARAMS, manipulation="magnitude", scale_c=3.0)
    d = generate_dataset(p)
    np.testing.assert_allclose(d.step_B, 3.0 * d.step_A, rtol=1e-12)


def test_step_orientation_length_preserving():
    p = replace(_CLEAN_PARAMS, manipulation="orientation", angle_theta=60.0)
    d = generate_dataset(p)
    assert np.isclose(np.linalg.norm(d.step_B), np.linalg.norm(d.step_A), rtol=1e-10)


def test_step_orientation_orthogonal_at_90():
    p = replace(_CLEAN_PARAMS, manipulation="orientation", angle_theta=90.0)
    d = generate_dataset(p)
    cosine = np.dot(d.step_A, d.step_B) / (
        np.linalg.norm(d.step_A) * np.linalg.norm(d.step_B)
    )
    assert abs(cosine) < 1e-8


# ---------------------------------------------------------------------------
# Param validation
# ---------------------------------------------------------------------------


def test_validate_bad_n_components():
    with pytest.raises(LinearRecoveryError, match="n_components"):
        generate_dataset(replace(_CLEAN_PARAMS, n_components=1))


def test_validate_n_components_exceeds_features():
    with pytest.raises(LinearRecoveryError, match="n_components"):
        generate_dataset(replace(_CLEAN_PARAMS, n_components=11, n_features=10))


def test_validate_bad_noise():
    with pytest.raises(LinearRecoveryError, match="noise_scale"):
        generate_dataset(replace(_CLEAN_PARAMS, noise_scale=0.0))


def test_validate_bad_manipulation():
    with pytest.raises(LinearRecoveryError, match="manipulation"):
        generate_dataset(replace(_CLEAN_PARAMS, manipulation="shape"))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 5.2  Clean-floor: production estimators on full PCA (k = p = 10)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def clean_floor_results():
    """Measure delta and angle for all three manipulations under a fixed seed."""
    results = {}
    for manip in ("none", "magnitude", "orientation"):
        params = replace(_CLEAN_PARAMS, manipulation=manip)
        dataset = generate_dataset(params)
        delta, angle, *_ = project_and_measure(dataset, params)
        results[manip] = {"delta": delta, "angle": angle}
    return results


def test_none_delta_near_zero(clean_floor_results):
    assert clean_floor_results["none"]["delta"] < 1.5


def test_none_angle_near_zero(clean_floor_results):
    assert clean_floor_results["none"]["angle"] < 8.0


def test_magnitude_moves_delta(clean_floor_results):
    # scale_c=2 → ‖step_B_latent‖ ≈ 2 × ‖step_A_latent‖ → delta ≈ signal_scale
    assert clean_floor_results["magnitude"]["delta"] > 5.0


def test_magnitude_angle_near_null(clean_floor_results):
    none_angle = clean_floor_results["none"]["angle"]
    mag_angle = clean_floor_results["magnitude"]["angle"]
    assert mag_angle < max(none_angle * 3, 8.0)


def test_orientation_moves_angle(clean_floor_results):
    # Full PCA (k=p) → angle ≈ theta = 45°
    assert clean_floor_results["orientation"]["angle"] > 30.0


def test_orientation_delta_near_null(clean_floor_results):
    none_delta = clean_floor_results["none"]["delta"]
    ori_delta = clean_floor_results["orientation"]["delta"]
    # orientation is length-preserving → delta should stay near the null floor
    assert ori_delta < max(none_delta * 3, 2.0)


# ---------------------------------------------------------------------------
# 5.3  Inverse design: round-trip assertions and output shapes
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pca_and_latent():
    """Shared PCA fit for inverse-design tests."""
    params = replace(_CLEAN_PARAMS, manipulation="none")
    dataset = generate_dataset(params)
    _delta, _angle, pca, _Y, Vk = project_and_measure(dataset, params)
    # Latent-space step for group A
    a_latent = pca.components_ @ dataset.step_A
    return pca, Vk, a_latent, params


def test_inverse_design_magnitude_shape(pca_and_latent):
    _pca, Vk, a_latent, params = pca_and_latent
    delta_x = inverse_design_magnitude(a_latent, Vk, params.scale_c)
    assert delta_x.shape == (params.n_features,)


def test_inverse_design_magnitude_roundtrip(pca_and_latent):
    _pca, Vk, a_latent, params = pca_and_latent
    c = params.scale_c
    delta_x = inverse_design_magnitude(a_latent, Vk, c)
    recovered = Vk.T @ delta_x
    expected = (c - 1.0) * a_latent
    np.testing.assert_allclose(recovered, expected, atol=1e-10)


def test_inverse_design_orientation_roundtrip(pca_and_latent):
    _pca, Vk, a_latent, params = pca_and_latent
    k = params.n_components
    R = givens_rotation(k, np.deg2rad(params.angle_theta))
    delta_x = inverse_design_orientation(a_latent, Vk, R)
    recovered = Vk.T @ delta_x
    expected = (R - np.eye(k)) @ a_latent
    np.testing.assert_allclose(recovered, expected, atol=1e-10)


def test_givens_rotation_orthogonal():
    R = givens_rotation(6, np.pi / 4)
    np.testing.assert_allclose(R @ R.T, np.eye(6), atol=1e-12)


def test_givens_rotation_length_preserving():
    R = givens_rotation(6, np.pi / 3)
    v = np.random.default_rng(0).standard_normal(6)
    np.testing.assert_allclose(np.linalg.norm(R @ v), np.linalg.norm(v), rtol=1e-12)


def test_inverse_design_bad_vk_raises():
    # Non-orthonormal Vk should trigger the round-trip guard
    k = 3
    p = 5
    rng = np.random.default_rng(0)
    Vk_bad = rng.standard_normal((p, k))  # not orthonormal
    a = rng.standard_normal(k)
    with pytest.raises(LinearRecoveryError, match="Round-trip"):
        inverse_design_magnitude(a, Vk_bad, 2.0)


# ---------------------------------------------------------------------------
# delta_x_summary smoke test
# ---------------------------------------------------------------------------


def test_delta_x_summary_nonzero():
    rng = np.random.default_rng(0)
    dx = rng.standard_normal(50)
    dx[5] = 100.0  # dominant feature
    summary = delta_x_summary(dx)
    assert summary["n_support"] >= 1
    assert summary["participation_ratio"] > 0
    assert 0 < summary["top3_mass"] <= 1.0


def test_delta_x_summary_zero():
    summary = delta_x_summary(np.zeros(20))
    assert summary["n_support"] == 0
    assert summary["participation_ratio"] == 0.0
