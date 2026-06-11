"""Tests for the Rung-1 methylation rev.logit recovery test bed."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from motco.simulations.generator import rev_logit
from motco.simulations.methylation_recovery import (
    MethylationRecoveryError,
    MethylationRecoveryParams,
    generate_dataset,
    givens_rotation,
    inverse_design_magnitude_mvalue,
    inverse_design_orientation_mvalue,
    jacobian_diag,
    project_and_measure,
    run_operating_point_sweep,
    run_step_scale_sweep,
)

# Regime with clean center recovery (reproduces the Rung-0 floor) and a small
# enough cross-talk signal that the nonlinearity, not the k-noise floor, is the
# variable under test.
_BASE = MethylationRecoveryParams(
    seed=0,
    n_features=50,
    n_samples_per_cell=40,
    noise_scale=0.3,
    signal_scale=2.0,
    n_components=2,
    m_baseline=0.0,
)


# ---------------------------------------------------------------------------
# 5.1  Deterministic output, structure, β range
# ---------------------------------------------------------------------------


def test_deterministic_same_seed():
    d1 = generate_dataset(_BASE)
    d2 = generate_dataset(replace(_BASE))
    np.testing.assert_array_equal(d1.X.to_numpy(), d2.X.to_numpy())


def test_different_seeds_differ():
    d1 = generate_dataset(_BASE)
    d2 = generate_dataset(replace(_BASE, seed=99))
    assert not np.allclose(d1.X.to_numpy(), d2.X.to_numpy())


def test_beta_in_unit_interval():
    d = generate_dataset(_BASE)
    x = d.X.to_numpy()
    assert x.min() > 0.0 and x.max() < 1.0


def test_two_stages_per_group():
    d = generate_dataset(_BASE)
    for group in ("A", "B"):
        stages = d.metadata.loc[d.metadata["group"] == group, "stage"].unique()
        assert set(stages) == {"0", "1"}


def test_feature_matrix_shape():
    d = generate_dataset(_BASE)
    n_total = 4 * _BASE.n_samples_per_cell
    assert d.X.shape == (n_total, _BASE.n_features)


def test_per_cell_beta_means_match_revlogit_of_spec():
    # Per-cell β means should approximate rev_logit of the M-space configuration.
    d = generate_dataset(_BASE)
    base = np.full(_BASE.n_features, _BASE.m_baseline)
    expected = {
        ("A", "0"): rev_logit(base),
        ("A", "1"): rev_logit(base + d.step_A),
        ("B", "0"): rev_logit(base),
        ("B", "1"): rev_logit(base + d.step_B),
    }
    for (group, stage), exp in expected.items():
        mask = (d.metadata["group"] == group) & (d.metadata["stage"] == stage)
        cell_mean = d.X.loc[mask.values].mean(axis=0).to_numpy()
        # Jensen gap + sampling error; tol generous but well under the signal.
        assert np.allclose(cell_mean, exp, atol=0.05)


# ---------------------------------------------------------------------------
# Param validation
# ---------------------------------------------------------------------------


def test_validate_bad_n_components():
    with pytest.raises(MethylationRecoveryError, match="n_components"):
        generate_dataset(replace(_BASE, n_components=1))


def test_validate_bad_noise():
    with pytest.raises(MethylationRecoveryError, match="noise_scale"):
        generate_dataset(replace(_BASE, noise_scale=0.0))


def test_validate_nonfinite_baseline():
    with pytest.raises(MethylationRecoveryError, match="m_baseline"):
        generate_dataset(replace(_BASE, m_baseline=float("inf")))


def test_validate_bad_manipulation():
    with pytest.raises(MethylationRecoveryError, match="manipulation"):
        generate_dataset(replace(_BASE, manipulation="shape"))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 5.2  Center operating point reduces to the Rung-0 clean floor
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def center_results():
    results = {}
    for manip in ("none", "magnitude", "orientation"):
        params = replace(_BASE, manipulation=manip, m_baseline=0.0)
        delta, angle, *_ = project_and_measure(generate_dataset(params), params)
        results[manip] = {"delta": delta, "angle": angle}
    return results


def test_center_magnitude_moves_delta(center_results):
    # magnitude registers a clear delta well above the null
    assert center_results["magnitude"]["delta"] > 10 * center_results["none"]["delta"]


def test_center_magnitude_angle_near_floor(center_results):
    # at center (small step), magnitude does NOT leak into angle
    assert center_results["magnitude"]["angle"] < center_results["none"]["angle"] + 5.0


def test_center_orientation_moves_angle(center_results):
    assert center_results["orientation"]["angle"] > 35.0


def test_center_orientation_delta_near_floor(center_results):
    assert center_results["orientation"]["delta"] < 5 * center_results["none"]["delta"] + 0.05


# ---------------------------------------------------------------------------
# 5.3  Tail operating point compresses magnitude delta (nonlinearity exercised)
# ---------------------------------------------------------------------------


def test_tail_compresses_magnitude_delta():
    center = replace(_BASE, manipulation="magnitude", m_baseline=0.0)
    tail = replace(_BASE, manipulation="magnitude", m_baseline=3.0)
    d_center, _, *_ = project_and_measure(generate_dataset(center), center)
    d_tail, _, *_ = project_and_measure(generate_dataset(tail), tail)
    # saturating baseline shrinks the measured magnitude delta
    assert d_tail < 0.6 * d_center


# ---------------------------------------------------------------------------
# 5.4  First-order inverse design round-trips near the center
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pca_and_latent():
    params = replace(_BASE, manipulation="none")
    _delta, _angle, pca, _Y, Vk = project_and_measure(generate_dataset(params), params)
    dataset = generate_dataset(params)
    a_latent = pca.components_ @ dataset.step_A
    return pca, Vk, a_latent


def test_jacobian_diag_center_is_quarter():
    j = jacobian_diag(0.0, 5)
    np.testing.assert_allclose(j, 0.25, atol=1e-12)


def test_first_order_magnitude_roundtrip_near_center(pca_and_latent):
    _pca, Vk, a_latent = pca_and_latent
    m_baseline = 0.0
    # small target so the linearization is accurate
    c = 1.05
    a_small = a_latent * 0.05 / max(np.linalg.norm(a_latent), 1e-12)
    delta_m = inverse_design_magnitude_mvalue(a_small, Vk, c, m_baseline)
    # nonlinear forward should recover the requested β-feature change to 1st order
    target_beta = Vk @ ((c - 1.0) * a_small)
    actual_beta = rev_logit(m_baseline + delta_m) - rev_logit(np.full_like(delta_m, m_baseline))
    np.testing.assert_allclose(actual_beta, target_beta, atol=1e-3)


def test_first_order_orientation_roundtrip_near_center(pca_and_latent):
    _pca, Vk, a_latent = pca_and_latent
    m_baseline = 0.0
    k = len(a_latent)
    R = givens_rotation(k, np.deg2rad(2.0))  # small rotation
    a_small = a_latent * 0.05 / max(np.linalg.norm(a_latent), 1e-12)
    delta_m = inverse_design_orientation_mvalue(a_small, Vk, R, m_baseline)
    target_beta = Vk @ ((R - np.eye(k)) @ a_small)
    actual_beta = rev_logit(m_baseline + delta_m) - rev_logit(np.full_like(delta_m, m_baseline))
    np.testing.assert_allclose(actual_beta, target_beta, atol=1e-3)


# ---------------------------------------------------------------------------
# Sweep drivers: shape and monotonicity smoke tests
# ---------------------------------------------------------------------------


def test_operating_point_sweep_columns():
    df = run_operating_point_sweep([0.0, 2.0], seeds=[0, 1], base_params=_BASE)
    assert set(df["manipulation"]) == {"none", "magnitude", "orientation"}
    for col in ("m_baseline", "beta_baseline", "slope", "delta_mean", "angle_mean"):
        assert col in df.columns
    # slope = β(1-β) decreases away from center
    centre = df.loc[df["m_baseline"] == 0.0, "slope"].iloc[0]
    tail = df.loc[df["m_baseline"] == 2.0, "slope"].iloc[0]
    assert tail < centre


def test_step_scale_sweep_magnitude_delta_monotone_then_saturates():
    df = run_step_scale_sweep([1.0, 2.0, 4.0], seeds=[0, 1, 2], base_params=_BASE)
    mag = df[df["manipulation"] == "magnitude"].sort_values("signal_scale")
    deltas = mag["delta_mean"].to_numpy()
    # delta grows with step scale (before plateau)
    assert deltas[1] > deltas[0]
    assert deltas[2] > deltas[1]
