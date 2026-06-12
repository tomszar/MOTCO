"""Tests for the Rung-2 projector recovery test bed."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from motco.simulations.projector_recovery import (
    ProjectorRecoveryError,
    ProjectorRecoveryParams,
    generate_dataset,
    project,
    project_and_measure,
    run_leakage_probe,
    run_projector_comparison,
)

# Clean linear floor: good SNR, signal in ≤ 2 dims, isotropic noise so the only
# variable under test is the projector. Mirrors the Rung-0 defaults.
_BASE = ProjectorRecoveryParams(
    seed=0,
    n_features=50,
    n_samples_per_cell=40,
    noise_scale=1.0,
    signal_scale=5.0,
    n_components=2,
)

_PROJECTORS = ["pca", "standardize", "plsda", "snf"]


# ---------------------------------------------------------------------------
# 4.1  Determinism, structure, reused geometry
# ---------------------------------------------------------------------------


def test_deterministic_same_seed():
    d1, a1, _ = project_and_measure(generate_dataset(_BASE), _BASE)
    d2, a2, _ = project_and_measure(generate_dataset(replace(_BASE)), _BASE)
    assert (d1, a1) == (d2, a2)


def test_deterministic_snf_same_seed():
    p = replace(_BASE, projector="snf", manipulation="magnitude", seed=3)
    r1 = project_and_measure(generate_dataset(p), p)[:2]
    r2 = project_and_measure(generate_dataset(p), p)[:2]
    assert r1 == r2


def test_different_seeds_differ():
    d1 = generate_dataset(_BASE)
    d2 = generate_dataset(replace(_BASE, seed=99))
    assert not np.allclose(d1.X.to_numpy(), d2.X.to_numpy())


def test_exactly_two_stages():
    d = generate_dataset(_BASE)
    assert sorted(d.metadata["stage"].unique().tolist()) == ["0", "1"]
    assert sorted(d.metadata["group"].unique().tolist()) == ["A", "B"]


def test_reuses_rung0_geometry():
    # step_A / step_B are the Rung-0 construction: magnitude scales A's step by c.
    p = replace(_BASE, manipulation="magnitude")
    d = generate_dataset(p)
    np.testing.assert_allclose(d.step_B, p.scale_c * d.step_A, rtol=1e-12)


# ---------------------------------------------------------------------------
# 4.2  PCA arm reproduces the Rung-0 clean floor
# ---------------------------------------------------------------------------


def test_pca_magnitude_moves_delta_only():
    none_d, none_a, _ = project_and_measure(
        generate_dataset(replace(_BASE, projector="pca", manipulation="none")),
        replace(_BASE, projector="pca", manipulation="none"),
    )
    p = replace(_BASE, projector="pca", manipulation="magnitude")
    d, a, _ = project_and_measure(generate_dataset(p), p)
    # delta well above the null floor; angle stays near the none floor.
    assert d > 10 * none_d
    assert a < none_a + 3.0


def test_pca_orientation_moves_angle_only():
    none = replace(_BASE, projector="pca", manipulation="none")
    none_d, _, _ = project_and_measure(generate_dataset(none), none)
    p = replace(_BASE, projector="pca", manipulation="orientation")
    d, a, _ = project_and_measure(generate_dataset(p), p)
    # angle recovers the injected 45° within tolerance; delta near the null floor.
    assert abs(a - _BASE.angle_theta) < 10.0
    assert d < none_d + 1.0


# ---------------------------------------------------------------------------
# 4.3  Projector contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("projector", _PROJECTORS)
@pytest.mark.parametrize("manipulation", ["none", "magnitude", "orientation"])
def test_projector_contract(projector, manipulation):
    p = replace(_BASE, projector=projector, manipulation=manipulation)
    ds = generate_dataset(p)
    Y = project(ds, p)
    n_samples = 4 * p.n_samples_per_cell
    assert Y.shape == (n_samples, p.n_components)
    assert np.all(np.isfinite(Y.to_numpy()))
    delta, angle, _ = project_and_measure(ds, p)
    assert np.isfinite(delta) and np.isfinite(angle)


def test_standardize_recovers_under_anisotropy():
    # Under heteroscedastic noise, raw PCA loses the signal to large-variance
    # noise directions, but per-feature standardization restores clean recovery.
    aniso = dict(anisotropy=1.5, manipulation="orientation")
    pca = replace(_BASE, projector="pca", **aniso)
    std = replace(_BASE, projector="standardize", **aniso)
    _, pca_angle, _ = project_and_measure(generate_dataset(pca), pca)
    _, std_angle, _ = project_and_measure(generate_dataset(std), std)
    # standardize recovers the injected orientation; raw PCA does not.
    assert abs(std_angle - _BASE.angle_theta) < 10.0
    assert abs(pca_angle - _BASE.angle_theta) > 30.0


# ---------------------------------------------------------------------------
# 4.4  Supervised-leakage probe
# ---------------------------------------------------------------------------


def test_leakage_probe_reports_both_references():
    probe = run_leakage_probe([2, 4], seeds=[0, 1, 2], base_params=_BASE)
    projectors = set(probe["projector"])
    assert projectors == {"pca", "plsda"}
    assert set(probe["n_components"]) == {2, 4}
    # Both references report a finite null delta/angle for each component count.
    assert np.all(np.isfinite(probe["delta_mean"]))
    assert np.all(np.isfinite(probe["angle_mean"]))


def test_projector_comparison_has_null_floor_and_reference():
    comp = run_projector_comparison(seeds=[0, 1, 2], base_params=_BASE)
    # one row per (projector, manipulation)
    assert len(comp) == len(_PROJECTORS) * 3
    assert set(comp["projector"]) == set(_PROJECTORS)
    # PCA magnitude separates cleanly from its own none floor (the reference).
    pca = comp[comp["projector"] == "pca"].set_index("manipulation")
    assert pca.loc["magnitude", "delta_mean"] > 5 * pca.loc["none", "delta_mean"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validation_rejects_bad_projector():
    with pytest.raises(ProjectorRecoveryError):
        generate_dataset(replace(_BASE, projector="bogus"))  # type: ignore[arg-type]


def test_validation_rejects_too_few_samples():
    with pytest.raises(ProjectorRecoveryError):
        generate_dataset(replace(_BASE, n_samples_per_cell=1))


def test_validation_rejects_components_over_features():
    with pytest.raises(ProjectorRecoveryError):
        generate_dataset(replace(_BASE, n_components=999))


def test_validation_rejects_negative_anisotropy():
    with pytest.raises(ProjectorRecoveryError):
        generate_dataset(replace(_BASE, anisotropy=-1.0))
