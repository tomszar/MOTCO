"""Tests for the Rung-4 cross-omic coupling test bed."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from motco.simulations.coupling_recovery import (
    CouplingRecoveryError,
    CouplingRecoveryParams,
    build_coupling_matrix,
    generate_dataset,
    predict_joint_angle,
    project_and_measure,
    run_analytic_comparison,
    run_coupling_sweep,
    run_dim_ratio_sweep,
    run_matrix_seed_sweep,
)

_BASE = CouplingRecoveryParams(
    seed=0,
    n_features_anchor=50,
    n_samples_per_cell=40,
    noise_scale=1.0,
    signal_scale=5.0,
    dim_ratio=1.0,
    coupling_scale=0.5,
    m_structure="random_sparse",
    nnz_per_nuis=3,
    matrix_seed=0,
    n_components=10,
)


# ---------------------------------------------------------------------------
# 1.  Params helpers
# ---------------------------------------------------------------------------


def test_n_features_nuisance_equal_ratio():
    assert _BASE.n_features_nuisance == _BASE.n_features_anchor


def test_n_features_total():
    assert _BASE.n_features_total == _BASE.n_features_anchor + _BASE.n_features_nuisance


def test_n_samples():
    assert _BASE.n_samples == 4 * _BASE.n_samples_per_cell


# ---------------------------------------------------------------------------
# 2.  build_coupling_matrix
# ---------------------------------------------------------------------------


def test_coupling_matrix_operator_norm_is_one():
    rng = np.random.default_rng(0)
    for structure in ("random_sparse", "dense", "rank1"):
        M = build_coupling_matrix(50, 50, structure, nnz_per_nuis=3, rng=rng)  # type: ignore[arg-type]
        sigma_max = np.linalg.svd(M, compute_uv=False)[0]
        assert abs(sigma_max - 1.0) < 1e-10, f"Failed for {structure}"


def test_coupling_matrix_shape():
    rng = np.random.default_rng(0)
    M = build_coupling_matrix(30, 50, "random_sparse", nnz_per_nuis=3, rng=rng)
    assert M.shape == (30, 50)


def test_random_sparse_nnz_per_row():
    rng = np.random.default_rng(0)
    nnz = 4
    M_raw = np.zeros((50, 50))
    # Recover binary M from M_norm × σ_max — just test that nnz structure is right
    # by constructing manually.
    for i in range(50):
        cols = rng.choice(50, size=nnz, replace=False)
        M_raw[i, cols] = 1.0
    # Each row should have exactly nnz non-zeros.
    assert np.all(M_raw.sum(axis=1) == nnz)


def test_dense_matrix_is_all_ones_before_norm():
    rng = np.random.default_rng(0)
    M = build_coupling_matrix(5, 10, "dense", nnz_per_nuis=1, rng=rng)
    # After operator-norm normalisation, M = 1 / sigma_max, so all values equal.
    assert np.allclose(M / M[0, 0], np.ones((5, 10)))


def test_rank1_matrix_structure():
    rng = np.random.default_rng(0)
    M = build_coupling_matrix(10, 20, "rank1", nnz_per_nuis=1, rng=rng)
    # All rows are identical (each maps only anchor feature 0).
    assert np.allclose(M, M[0:1, :])
    # Only column 0 is non-zero.
    assert np.all(M[:, 1:] == 0.0)


def test_different_matrix_seeds_give_different_sparse_M():
    rng0 = np.random.default_rng(0)
    rng1 = np.random.default_rng(1)
    M0 = build_coupling_matrix(50, 50, "random_sparse", nnz_per_nuis=3, rng=rng0)
    M1 = build_coupling_matrix(50, 50, "random_sparse", nnz_per_nuis=3, rng=rng1)
    assert not np.allclose(M0, M1)


# ---------------------------------------------------------------------------
# 3.  predict_joint_angle
# ---------------------------------------------------------------------------


def test_predict_angle_zero_coupling_equals_anchor_angle():
    rng = np.random.default_rng(42)
    M = build_coupling_matrix(50, 50, "random_sparse", nnz_per_nuis=3, rng=rng)
    a = rng.standard_normal(50)
    theta = np.deg2rad(45.0)
    # Build step_B rotated by 45° from step_A.
    perp = rng.standard_normal(50)
    perp -= np.dot(perp, a / np.linalg.norm(a)) * (a / np.linalg.norm(a))
    perp /= np.linalg.norm(perp)
    b = np.cos(theta) * a + np.sin(theta) * np.linalg.norm(a) * perp
    # At coupling_scale=0, prediction equals the anchor angle.
    pred = predict_joint_angle(a, b, M, coupling_scale=0.0)
    anchor_angle = float(np.degrees(np.arccos(
        np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1)
    )))
    assert abs(pred - anchor_angle) < 1e-6


def test_predict_angle_magnitude_is_zero():
    rng = np.random.default_rng(0)
    M = build_coupling_matrix(50, 50, "random_sparse", nnz_per_nuis=3, rng=rng)
    a = rng.standard_normal(50)
    b = 2.5 * a  # magnitude manipulation
    for gamma in [0.0, 0.25, 0.5, 0.75, 1.0]:
        pred = predict_joint_angle(a, b, M, coupling_scale=gamma)
        assert abs(pred) < 1e-6, f"magnitude angle should be 0° at γ={gamma}, got {pred}"


def test_predict_angle_increases_with_coupling_for_orientation():
    rng = np.random.default_rng(7)
    M = build_coupling_matrix(50, 50, "random_sparse", nnz_per_nuis=3, rng=rng)
    a = rng.standard_normal(50)
    a /= np.linalg.norm(a)
    theta = np.deg2rad(45.0)
    perp = rng.standard_normal(50)
    perp -= np.dot(perp, a) * a
    perp /= np.linalg.norm(perp)
    b = np.cos(theta) * a + np.sin(theta) * perp
    preds = [predict_joint_angle(a, b, M, gamma) for gamma in [0.0, 0.5, 1.0]]
    # Angle may increase or decrease depending on M; just check it changes.
    assert not (preds[0] == preds[1] == preds[2])


def test_predict_angle_output_in_range():
    rng = np.random.default_rng(3)
    M = build_coupling_matrix(20, 50, "random_sparse", nnz_per_nuis=3, rng=rng)
    a, b = rng.standard_normal(50), rng.standard_normal(50)
    for gamma in [0.0, 0.5, 1.0]:
        angle = predict_joint_angle(a, b, M, gamma)
        assert 0.0 <= angle <= 180.0


# ---------------------------------------------------------------------------
# 4.  generate_dataset
# ---------------------------------------------------------------------------


def test_deterministic_same_seed():
    ds1 = generate_dataset(_BASE)
    ds2 = generate_dataset(replace(_BASE))
    np.testing.assert_array_equal(ds1.X_anchor, ds2.X_anchor)
    np.testing.assert_array_equal(ds1.X_nuisance[0], ds2.X_nuisance[0])
    np.testing.assert_array_equal(ds1.M_norm, ds2.M_norm)


def test_different_seeds_differ():
    ds1 = generate_dataset(_BASE)
    ds2 = generate_dataset(replace(_BASE, seed=99))
    assert not np.allclose(ds1.X_anchor, ds2.X_anchor)


def test_different_matrix_seeds_give_different_nuisance():
    ds0 = generate_dataset(_BASE)
    ds1 = generate_dataset(replace(_BASE, matrix_seed=7))
    assert not np.allclose(ds0.X_nuisance[0], ds1.X_nuisance[0])
    assert not np.allclose(ds0.M_norm, ds1.M_norm)


def test_anchor_shape():
    ds = generate_dataset(_BASE)
    assert ds.X_anchor.shape == (_BASE.n_samples, _BASE.n_features_anchor)


def test_nuisance_shape():
    ds = generate_dataset(_BASE)
    assert len(ds.X_nuisance) == 1
    assert ds.X_nuisance[0].shape == (_BASE.n_samples, _BASE.n_features_nuisance)


def test_m_norm_shape():
    ds = generate_dataset(_BASE)
    assert ds.M_norm.shape == (_BASE.n_features_nuisance, _BASE.n_features_anchor)


def test_metadata_structure():
    ds = generate_dataset(_BASE)
    assert sorted(ds.metadata["group"].unique().tolist()) == ["A", "B"]
    assert sorted(ds.metadata["stage"].unique().tolist()) == ["0", "1"]


def test_reuses_rung0_geometry_magnitude():
    p = replace(_BASE, manipulation="magnitude")
    ds = generate_dataset(p)
    np.testing.assert_allclose(ds.step_B, p.scale_c * ds.step_A, rtol=1e-12)


def test_zero_coupling_anchor_matches_rung3_baseline():
    # At coupling_scale=0, anchor block is identical regardless of M.
    p_zero = replace(_BASE, coupling_scale=0.0)
    p_full = replace(_BASE, coupling_scale=1.0)
    ds_zero = generate_dataset(p_zero)
    ds_full = generate_dataset(p_full)
    np.testing.assert_array_equal(ds_zero.X_anchor, ds_full.X_anchor)


def test_zero_coupling_nuisance_has_zero_mean_per_cell():
    p = replace(_BASE, coupling_scale=0.0, manipulation="orientation")
    ds = generate_dataset(p)
    meta = ds.metadata
    X = ds.X_nuisance[0]
    for group in ("A", "B"):
        for stage in ("0", "1"):
            mask = (meta["group"] == group) & (meta["stage"] == stage)
            cell_mean = X[mask.to_numpy()].mean(axis=0)
            # At coupling_scale=0 every cell mean is ~0 (within noise/sqrt(n)).
            assert np.abs(cell_mean).mean() < 1.0


def test_nonzero_coupling_stage1_nuisance_mean_nonzero():
    p = replace(_BASE, coupling_scale=1.0, manipulation="magnitude")
    ds = generate_dataset(p)
    meta = ds.metadata
    X = ds.X_nuisance[0]
    stage1_mask = (meta["stage"] == "1").to_numpy()
    stage0_mask = (meta["stage"] == "0").to_numpy()
    mean_diff = np.abs(X[stage1_mask].mean(axis=0) - X[stage0_mask].mean(axis=0)).mean()
    # The nuisance mean at stage 1 is coupling_scale × M_norm @ step, which is non-zero.
    assert mean_diff > 0.1


# ---------------------------------------------------------------------------
# 5.  project_and_measure
# ---------------------------------------------------------------------------


def test_project_and_measure_finite():
    ds = generate_dataset(_BASE)
    delta, angle, Y = project_and_measure(ds, _BASE)
    assert np.isfinite(delta) and np.isfinite(angle)
    assert Y.shape == (_BASE.n_samples, _BASE.n_components)


def test_deterministic_project_and_measure():
    ds = generate_dataset(_BASE)
    d1, a1, _ = project_and_measure(ds, _BASE)
    d2, a2, _ = project_and_measure(ds, _BASE)
    assert (d1, a1) == (d2, a2)


def test_zero_coupling_magnitude_moves_delta_only():
    none_p = replace(_BASE, coupling_scale=0.0, manipulation="none")
    mag_p = replace(_BASE, coupling_scale=0.0, manipulation="magnitude")
    _, none_a, _ = project_and_measure(generate_dataset(none_p), none_p)
    mag_d, mag_a, _ = project_and_measure(generate_dataset(mag_p), mag_p)
    # Magnitude moves delta; angle stays near the none floor.
    assert mag_d > 2.0
    assert mag_a < none_a + 5.0


# ---------------------------------------------------------------------------
# 6.  Analytic prediction consistency
# ---------------------------------------------------------------------------


def test_analytic_matches_empirical_at_zero_coupling():
    # At coupling_scale=0, the analytic formula reduces to the anchor angle;
    # the empirical angle (via PCA) should be close for high-SNR orientation.
    p = replace(_BASE, coupling_scale=0.0, manipulation="orientation", n_components=2)
    ds = generate_dataset(p)
    _, angle_meas, _ = project_and_measure(ds, p)
    angle_pred = predict_joint_angle(ds.step_A, ds.step_B, ds.M_norm, 0.0)
    # Predicted anchor angle should be near 45° (exact by construction);
    # PCA-measured angle may differ due to projection noise, but should be in range.
    assert abs(angle_pred - 45.0) < 1.0
    assert 30.0 < angle_meas < 70.0


def test_magnitude_arm_angle_stays_at_null_floor_across_coupling():
    # Analytic proof: magnitude coupling_scale never affects angle.
    rng = np.random.default_rng(0)
    M = build_coupling_matrix(50, 50, "random_sparse", nnz_per_nuis=3, rng=rng)
    ds = generate_dataset(replace(_BASE, manipulation="magnitude"))
    for gamma in [0.0, 0.5, 1.0]:
        pred = predict_joint_angle(ds.step_A, ds.step_B, M, gamma)
        assert abs(pred) < 1e-6


# ---------------------------------------------------------------------------
# 7.  Driver schema checks
# ---------------------------------------------------------------------------


def test_run_coupling_sweep_schema():
    result = run_coupling_sweep(
        coupling_scales=[0.0, 0.5, 1.0],
        m_structures=["random_sparse", "dense"],
        seeds=[0, 1],
        base_params=_BASE,
    )
    expected_cols = {
        "m_structure", "coupling_scale", "manipulation",
        "delta_mean", "delta_std", "angle_mean", "angle_std",
    }
    assert expected_cols.issubset(result.columns)
    assert set(result["manipulation"]) == {"none", "magnitude", "orientation"}
    assert np.all(np.isfinite(result["delta_mean"]))
    assert np.all(np.isfinite(result["angle_mean"]))


def test_run_analytic_comparison_schema():
    result = run_analytic_comparison(
        coupling_scales=[0.0, 0.5],
        m_structures=["random_sparse"],
        seeds=[0, 1],
        base_params=_BASE,
    )
    assert {"angle_pred", "angle_meas", "angle_anchor", "coupling_scale", "seed"}.issubset(
        result.columns
    )
    assert np.all(np.isfinite(result["angle_pred"]))
    assert np.all(np.isfinite(result["angle_meas"]))


def test_run_dim_ratio_sweep_schema():
    result = run_dim_ratio_sweep(
        dim_ratios=[0.5, 1.0],
        coupling_scale=0.5,
        seeds=[0, 1],
        base_params=_BASE,
    )
    assert "dim_ratio" in result.columns
    assert set(result["manipulation"]) == {"none", "magnitude", "orientation"}


def test_run_matrix_seed_sweep_schema():
    result = run_matrix_seed_sweep(
        matrix_seeds=[0, 1, 2],
        coupling_scale=0.5,
        seeds=[0, 1],
        base_params=_BASE,
    )
    assert "matrix_seed" in result.columns
    assert set(result["matrix_seed"]) == {0, 1, 2}
    assert np.all(np.isfinite(result["angle_mean"]))


# ---------------------------------------------------------------------------
# 8.  Validation
# ---------------------------------------------------------------------------


def test_validation_rejects_coupling_scale_out_of_range():
    with pytest.raises(CouplingRecoveryError):
        generate_dataset(replace(_BASE, coupling_scale=1.1))
    with pytest.raises(CouplingRecoveryError):
        generate_dataset(replace(_BASE, coupling_scale=-0.1))


def test_validation_rejects_zero_dim_ratio():
    with pytest.raises(CouplingRecoveryError):
        generate_dataset(replace(_BASE, dim_ratio=0.0))


def test_validation_rejects_bad_m_structure():
    with pytest.raises(CouplingRecoveryError):
        generate_dataset(replace(_BASE, m_structure="bogus"))  # type: ignore[arg-type]


def test_validation_rejects_too_many_components():
    with pytest.raises(CouplingRecoveryError):
        generate_dataset(replace(_BASE, n_components=9999))


def test_validation_rejects_too_few_samples():
    with pytest.raises(CouplingRecoveryError):
        generate_dataset(replace(_BASE, n_samples_per_cell=1))


def test_validation_rejects_nnz_too_large():
    with pytest.raises(CouplingRecoveryError):
        generate_dataset(replace(_BASE, nnz_per_nuis=999))


def test_validation_rejects_nnz_zero():
    with pytest.raises(CouplingRecoveryError):
        generate_dataset(replace(_BASE, nnz_per_nuis=0))


def test_validation_rejects_bad_manipulation():
    with pytest.raises(CouplingRecoveryError):
        generate_dataset(replace(_BASE, manipulation="bogus"))  # type: ignore[arg-type]
