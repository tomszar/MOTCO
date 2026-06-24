"""Tests for the Rung-3 multi-block concatenation test bed."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from motco.simulations.multiblock_recovery import (
    MultiblockRecoveryError,
    MultiblockRecoveryParams,
    build_joint_matrix,
    decompose_block_weight,
    generate_dataset,
    project_and_measure,
    run_block_comparison,
    run_block_weight_curve,
    run_dim_ratio_sweep,
    run_rho_sweep,
)

# Clean linear floor: high SNR, signal in ≤ 2 dims, one nuisance block at equal
# dimensionality.  Matches the Rung-2 baseline settings.
_BASE = MultiblockRecoveryParams(
    seed=0,
    n_features_anchor=50,
    n_samples_per_cell=40,
    noise_scale=1.0,
    signal_scale=5.0,
    n_nuisance_blocks=1,
    dim_ratio=1.0,
    rho_nuisance=0.0,
    n_components=10,
)


# ---------------------------------------------------------------------------
# 1.  Params helpers
# ---------------------------------------------------------------------------


def test_n_features_nuisance_equal_ratio():
    p = replace(_BASE, dim_ratio=1.0)
    assert p.n_features_nuisance == p.n_features_anchor


def test_n_features_nuisance_half_ratio():
    p = replace(_BASE, dim_ratio=0.5)
    assert p.n_features_nuisance == max(1, round(0.5 * p.n_features_anchor))


def test_n_features_total_one_block():
    p = replace(_BASE, n_nuisance_blocks=1, dim_ratio=1.0)
    assert p.n_features_total == p.n_features_anchor + p.n_features_nuisance


def test_n_features_total_two_blocks():
    p = replace(_BASE, n_nuisance_blocks=2, dim_ratio=2.0)
    assert p.n_features_total == p.n_features_anchor + 2 * p.n_features_nuisance


def test_n_features_nuisance_zero_when_no_blocks():
    p = replace(_BASE, n_nuisance_blocks=0)
    assert p.n_features_nuisance == 0
    assert p.n_features_total == p.n_features_anchor


def test_n_samples():
    assert _BASE.n_samples == 4 * _BASE.n_samples_per_cell


# ---------------------------------------------------------------------------
# 2.  Determinism and structure
# ---------------------------------------------------------------------------


def test_deterministic_same_seed():
    ds1 = generate_dataset(_BASE)
    ds2 = generate_dataset(replace(_BASE))
    np.testing.assert_array_equal(ds1.X_anchor, ds2.X_anchor)
    np.testing.assert_array_equal(ds1.X_nuisance[0], ds2.X_nuisance[0])


def test_different_seeds_differ():
    ds1 = generate_dataset(_BASE)
    ds2 = generate_dataset(replace(_BASE, seed=99))
    assert not np.allclose(ds1.X_anchor, ds2.X_anchor)


def test_metadata_has_correct_groups_and_stages():
    ds = generate_dataset(_BASE)
    assert sorted(ds.metadata["group"].unique().tolist()) == ["A", "B"]
    assert sorted(ds.metadata["stage"].unique().tolist()) == ["0", "1"]


def test_anchor_shape():
    ds = generate_dataset(_BASE)
    assert ds.X_anchor.shape == (_BASE.n_samples, _BASE.n_features_anchor)


def test_nuisance_shape_one_block():
    ds = generate_dataset(_BASE)
    assert len(ds.X_nuisance) == 1
    assert ds.X_nuisance[0].shape == (_BASE.n_samples, _BASE.n_features_nuisance)


def test_nuisance_shape_two_blocks():
    p = replace(_BASE, n_nuisance_blocks=2, dim_ratio=2.0)
    ds = generate_dataset(p)
    assert len(ds.X_nuisance) == 2
    for X in ds.X_nuisance:
        assert X.shape == (p.n_samples, p.n_features_nuisance)


def test_no_nuisance_blocks_gives_empty_list():
    p = replace(_BASE, n_nuisance_blocks=0)
    ds = generate_dataset(p)
    assert ds.X_nuisance == []


def test_reuses_rung0_geometry_magnitude():
    p = replace(_BASE, manipulation="magnitude")
    ds = generate_dataset(p)
    np.testing.assert_allclose(ds.step_B, p.scale_c * ds.step_A, rtol=1e-12)


def test_anchor_block_matches_single_block_baseline():
    # With n_nuisance_blocks=0 the anchor block matches the Rung-2 standardize
    # baseline exactly (same RNG sequence: seed → 4 cells of anchor noise).
    single = replace(_BASE, n_nuisance_blocks=0)
    multi = replace(_BASE, n_nuisance_blocks=1)
    ds_single = generate_dataset(single)
    ds_multi = generate_dataset(multi)
    np.testing.assert_array_equal(ds_single.X_anchor, ds_multi.X_anchor)


# ---------------------------------------------------------------------------
# 3.  Nuisance block correlation
# ---------------------------------------------------------------------------


def test_rho_zero_gives_independent_nuisance():
    p = replace(_BASE, rho_nuisance=0.0)
    ds = generate_dataset(p)
    X = ds.X_nuisance[0]
    # Row-wise correlations should be low for independent features.
    C = np.corrcoef(X.T)
    off_diag = C[np.triu_indices_from(C, k=1)]
    assert np.abs(off_diag).mean() < 0.3


def test_rho_positive_increases_feature_correlation():
    p_low = replace(_BASE, rho_nuisance=0.0, seed=1)
    p_high = replace(_BASE, rho_nuisance=0.7, seed=1)
    ds_low = generate_dataset(p_low)
    ds_high = generate_dataset(p_high)

    def mean_off_diag_corr(X: np.ndarray) -> float:
        C = np.corrcoef(X.T)
        off = C[np.triu_indices_from(C, k=1)]
        return float(np.abs(off).mean())

    assert mean_off_diag_corr(ds_high.X_nuisance[0]) > mean_off_diag_corr(
        ds_low.X_nuisance[0]
    )


# ---------------------------------------------------------------------------
# 4.  Joint matrix and block-weight decomposition
# ---------------------------------------------------------------------------


def test_joint_matrix_shape():
    ds = generate_dataset(_BASE)
    X_joint, p_a = build_joint_matrix(ds)
    assert X_joint.shape == (_BASE.n_samples, _BASE.n_features_total)
    assert p_a == _BASE.n_features_anchor


def test_joint_matrix_per_block_zero_mean():
    ds = generate_dataset(_BASE)
    X_joint, p_a = build_joint_matrix(ds)
    # Each block is independently standardised, so the mean of each column
    # of the joint matrix is zero (within floating-point tolerance).
    np.testing.assert_allclose(X_joint.mean(axis=0), 0.0, atol=1e-10)


def test_joint_matrix_per_block_unit_std():
    ds = generate_dataset(_BASE)
    X_joint, _ = build_joint_matrix(ds)
    # Each column's std should be 1 after per-block standardisation.
    np.testing.assert_allclose(X_joint.std(axis=0), 1.0, atol=1e-10)


def test_single_block_joint_matrix_identity():
    p = replace(_BASE, n_nuisance_blocks=0)
    ds = generate_dataset(p)
    X_joint, p_a = build_joint_matrix(ds)
    assert X_joint.shape == (_BASE.n_samples, _BASE.n_features_anchor)
    assert p_a == _BASE.n_features_anchor


def test_block_weight_is_one_for_single_block():
    p = replace(_BASE, n_nuisance_blocks=0)
    ds = generate_dataset(p)
    X_joint, p_a = build_joint_matrix(ds)
    w, _ = decompose_block_weight(X_joint, p_a, p.n_components)
    assert abs(w - 1.0) < 1e-10


def test_block_weight_decreases_with_dim_ratio():
    # As nuisance dimensionality grows, the anchor should capture less loading.
    p_low = replace(_BASE, dim_ratio=1.0)
    p_high = replace(_BASE, dim_ratio=5.0)

    def get_w(params: MultiblockRecoveryParams) -> float:
        ds = generate_dataset(params)
        X_joint, p_a = build_joint_matrix(ds)
        w, _ = decompose_block_weight(X_joint, p_a, params.n_components)
        return w

    assert get_w(p_low) > get_w(p_high)


def test_block_weight_per_component_shape():
    ds = generate_dataset(_BASE)
    X_joint, p_a = build_joint_matrix(ds)
    w, per_comp = decompose_block_weight(X_joint, p_a, _BASE.n_components)
    assert per_comp.shape == (_BASE.n_components,)
    assert np.all(per_comp >= 0)
    assert np.all(per_comp <= 1.0 + 1e-10)


def test_block_weight_between_zero_and_one():
    ds = generate_dataset(_BASE)
    X_joint, p_a = build_joint_matrix(ds)
    w, _ = decompose_block_weight(X_joint, p_a, _BASE.n_components)
    assert 0.0 < w < 1.0


# ---------------------------------------------------------------------------
# 5.  project_and_measure — contract and clean-floor checks
# ---------------------------------------------------------------------------


def test_project_and_measure_finite():
    ds = generate_dataset(_BASE)
    delta, angle, Y = project_and_measure(ds, _BASE)
    assert np.isfinite(delta)
    assert np.isfinite(angle)
    assert Y.shape == (_BASE.n_samples, _BASE.n_components)


def test_deterministic_project_and_measure():
    ds = generate_dataset(_BASE)
    d1, a1, _ = project_and_measure(ds, _BASE)
    d2, a2, _ = project_and_measure(ds, _BASE)
    assert (d1, a1) == (d2, a2)


def test_single_block_magnitude_moves_delta_only():
    # Single-block floor should reproduce the Rung-2 standardize clean result.
    none_p = replace(_BASE, n_nuisance_blocks=0, manipulation="none")
    mag_p = replace(_BASE, n_nuisance_blocks=0, manipulation="magnitude")
    none_d, none_a, _ = project_and_measure(generate_dataset(none_p), none_p)
    mag_d, mag_a, _ = project_and_measure(generate_dataset(mag_p), mag_p)
    assert mag_d > 5 * none_d
    assert mag_a < none_a + 5.0


def test_single_block_orientation_moves_angle_only():
    none_p = replace(_BASE, n_nuisance_blocks=0, manipulation="none")
    ori_p = replace(_BASE, n_nuisance_blocks=0, manipulation="orientation")
    none_d, _, _ = project_and_measure(generate_dataset(none_p), none_p)
    ori_d, ori_a, _ = project_and_measure(generate_dataset(ori_p), ori_p)
    assert abs(ori_a - _BASE.angle_theta) < 10.0
    assert ori_d < none_d + 1.0


# ---------------------------------------------------------------------------
# 6.  Driver outputs — schema and structure
# ---------------------------------------------------------------------------


def test_run_block_comparison_schema():
    result = run_block_comparison(
        seeds=[0, 1],
        base_params=_BASE,
        dim_ratios=[1.0, 5.0],
    )
    expected_cols = {
        "n_nuisance_blocks", "dim_ratio", "manipulation",
        "delta_mean", "delta_std", "angle_mean", "angle_std",
    }
    assert expected_cols.issubset(result.columns)
    # Single-block baseline (n_nuisance_blocks=0) + 2 blocks × 2 dim_ratios × 3 manips
    assert 0 in result["n_nuisance_blocks"].values
    assert set(result["manipulation"]) == {"none", "magnitude", "orientation"}
    assert np.all(np.isfinite(result["delta_mean"]))
    assert np.all(np.isfinite(result["angle_mean"]))


def test_run_dim_ratio_sweep_schema():
    result = run_dim_ratio_sweep(
        dim_ratios=[1.0, 5.0],
        seeds=[0, 1],
        base_params=_BASE,
    )
    assert set(result.columns) >= {"dim_ratio", "manipulation", "delta_mean", "angle_mean"}
    # Includes the single-block baseline (dim_ratio=0.0) and two swept values.
    assert 0.0 in result["dim_ratio"].values
    assert 1.0 in result["dim_ratio"].values


def test_run_rho_sweep_schema():
    result = run_rho_sweep(
        rho_values=[0.0, 0.7],
        dim_ratio=5.0,
        seeds=[0, 1],
        base_params=_BASE,
    )
    assert "rho_nuisance" in result.columns
    assert set(result["rho_nuisance"]) == {0.0, 0.7}
    assert np.all(np.isfinite(result["angle_mean"]))


def test_run_block_weight_curve_schema():
    result = run_block_weight_curve(
        dim_ratios=[1.0, 5.0],
        n_nuisance_blocks=1,
        seeds=[0, 1],
        base_params=_BASE,
    )
    assert "dim_ratio" in result.columns
    assert "w_anchor" in result.columns
    assert "p_anchor_naive" in result.columns
    assert np.all(result["w_anchor"] >= 0.0)
    assert np.all(result["w_anchor"] <= 1.0 + 1e-10)
    # Baseline row (dim_ratio=0) has w_anchor=1.
    baseline = result[result["dim_ratio"] == 0.0]
    assert np.all(baseline["w_anchor"] == 1.0)


def test_block_weight_curve_naive_fraction():
    result = run_block_weight_curve(
        dim_ratios=[1.0],
        n_nuisance_blocks=1,
        seeds=[0],
        base_params=_BASE,
    )
    # At dim_ratio=1, p_nuisance = p_anchor, so p_anchor_naive = 0.5.
    row = result[result["dim_ratio"] == 1.0].iloc[0]
    assert abs(row["p_anchor_naive"] - 0.5) < 1e-10


# ---------------------------------------------------------------------------
# 7.  Validation
# ---------------------------------------------------------------------------


def test_validation_rejects_zero_dim_ratio_with_nuisance():
    with pytest.raises(MultiblockRecoveryError):
        generate_dataset(replace(_BASE, n_nuisance_blocks=1, dim_ratio=0.0))


def test_validation_rejects_negative_nuisance_blocks():
    with pytest.raises(MultiblockRecoveryError):
        generate_dataset(replace(_BASE, n_nuisance_blocks=-1))


def test_validation_rejects_rho_out_of_range():
    with pytest.raises(MultiblockRecoveryError):
        generate_dataset(replace(_BASE, rho_nuisance=1.0))
    with pytest.raises(MultiblockRecoveryError):
        generate_dataset(replace(_BASE, rho_nuisance=-0.1))


def test_validation_rejects_too_many_components():
    with pytest.raises(MultiblockRecoveryError):
        generate_dataset(replace(_BASE, n_components=9999))


def test_validation_rejects_too_few_samples():
    with pytest.raises(MultiblockRecoveryError):
        generate_dataset(replace(_BASE, n_samples_per_cell=1))


def test_validation_rejects_bad_manipulation():
    with pytest.raises(MultiblockRecoveryError):
        generate_dataset(replace(_BASE, manipulation="bogus"))  # type: ignore[arg-type]
