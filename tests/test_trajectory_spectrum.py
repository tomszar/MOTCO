"""Latent stage-mean configuration spectrum: definition, opt-in, and inertness.

The relative eigengap is a *recorded covariate*. Two properties are load-bearing
and pinned here: it means what the geometry audit measured (a straight
trajectory saturates it, an isotropic one drives it to zero, and on a balanced
design it agrees with the audit's pooled-sample-mean formula), and switching it
on changes nothing about the test — same draws, same statistics, same p-values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from motco.stats.design import build_ls_means, get_model_matrix
from motco.stats.permutation import RRPP
from motco.stats.trajectory import (
    CONFIG_SPECTRUM_VERSION,
    configuration_spectra,
    configuration_spectrum,
    estimate_betas,
    estimate_difference,
    pooled_relative_eigengap,
)


def audit_relgap(configuration: np.ndarray) -> float:
    """The audit's reproduction snippet, verbatim in spirit.

    ``docs/reports/geometry-audit-2026-09-01.md``, experiment 2.
    """

    X = configuration - configuration.mean(0, keepdims=True)
    e = np.linalg.svd(X, compute_uv=False) ** 2
    return float((e[0] - e[1]) / e.sum())


def make_design(n_per_cell: int = 6, n_stages: int = 4, n_features: int = 5):
    """A balanced two-group design with a linear-plus-noise trajectory."""

    rng = np.random.default_rng(11)
    rows = []
    values = []
    for group in ("A", "B"):
        direction = np.linspace(1.0, 0.2, n_features) * (1.0 if group == "A" else 0.6)
        for stage in range(n_stages):
            for _ in range(n_per_cell):
                rows.append((group, f"t{stage}"))
                values.append(direction * stage + rng.normal(scale=0.15, size=n_features))
    frame = pd.DataFrame(rows, columns=["group", "stage"])
    Y = np.asarray(values, dtype=float)
    model_full = get_model_matrix(frame, group_col="group", level_col="stage", full=True)
    model_reduced = get_model_matrix(frame, group_col="group", level_col="stage", full=False)
    ls_means = build_ls_means(["A", "B"], [f"t{i}" for i in range(n_stages)], full=True)
    contrast = [list(range(n_stages)), list(range(n_stages, 2 * n_stages))]
    return frame, Y, model_full, model_reduced, ls_means, contrast


# ── 1.1 Definition ────────────────────────────────────────────────────────────


def test_straight_trajectory_saturates_the_eigengap() -> None:
    configuration = np.outer(np.arange(5.0), np.array([1.0, -2.0, 0.5]))

    entry = configuration_spectrum(configuration)

    assert entry["relative_eigengap"] == pytest.approx(1.0)
    assert entry["spectrum"][0] == pytest.approx(1.0)
    assert sum(entry["spectrum"]) == pytest.approx(1.0)


def test_isotropic_simplex_drives_the_eigengap_to_zero() -> None:
    """A regular simplex has one eigenvalue repeated: no axis dominates."""

    # Rows of a scaled identity, centered, are the vertices of a regular simplex
    # in the hyperplane orthogonal to the all-ones vector.
    configuration = np.eye(4)

    entry = configuration_spectrum(configuration)

    assert entry["relative_eigengap"] == pytest.approx(0.0, abs=1e-12)


def test_two_stage_configuration_is_exactly_saturated() -> None:
    """Rank at most 1 after centering, so the eigengap is 1 by construction."""

    entry = configuration_spectrum(np.array([[0.0, 0.0, 0.0], [3.0, -1.0, 2.0]]))

    assert entry["n_points"] == 2
    assert entry["relative_eigengap"] == 1.0


def test_zero_variance_configuration_records_an_explicit_sentinel() -> None:
    entry = configuration_spectrum(np.tile([1.0, 2.0], (4, 1)))

    assert entry["relative_eigengap"] is None
    assert entry["spectrum"] == []
    assert entry["total_variance"] == 0.0


def test_spectrum_matches_an_independent_svd() -> None:
    rng = np.random.default_rng(3)
    configuration = rng.normal(size=(5, 7))

    entry = configuration_spectrum(configuration)

    centered = configuration - configuration.mean(axis=0, keepdims=True)
    eigenvalues = np.linalg.svd(centered, compute_uv=False) ** 2
    np.testing.assert_allclose(entry["spectrum"], eigenvalues / eigenvalues.sum())
    assert entry["relative_eigengap"] == pytest.approx(audit_relgap(configuration))
    assert entry["total_variance"] == pytest.approx(float(eigenvalues.sum()))


def test_per_group_and_pooled_configurations_come_from_the_contrast() -> None:
    obs_vect = np.arange(24.0).reshape(8, 3)
    contrast = [[0, 1, 2, 3], [4, 5, 6, 7]]

    spectra = configuration_spectra(obs_vect, contrast)

    assert spectra["version"] == CONFIG_SPECTRUM_VERSION
    assert len(spectra["groups"]) == 2
    for index, levels in enumerate(contrast):
        expected = configuration_spectrum(obs_vect[levels, :])
        assert spectra["groups"][index] == expected
    pooled_configuration = (obs_vect[contrast[0], :] + obs_vect[contrast[1], :]) / 2.0
    assert spectra["pooled"] == configuration_spectrum(pooled_configuration)


def test_pooling_is_refused_when_groups_carry_different_stage_counts() -> None:
    spectra = configuration_spectra(np.arange(15.0).reshape(5, 3), [[0, 1], [2, 3, 4]])

    assert spectra["pooled"] is None
    assert pooled_relative_eigengap(spectra) is None
    assert len(spectra["groups"]) == 2


# ── 1.2 The audit-definition mirror ───────────────────────────────────────────


def test_pooled_ls_mean_eigengap_matches_the_audits_sample_mean_formula() -> None:
    """On a balanced design the two definitions coincide.

    The audit measured the eigengap from *pooled sample stage means*; the
    harness measures it from the pooled LS-mean rows, because those exist per
    permutation. Under the study's balanced designs they are the same
    configuration, which is what carries the audit's evidence over.
    """

    frame, Y, model_full, _, ls_means, contrast = make_design()

    _, _, _, spectra = estimate_difference(
        Y, model_full, ls_means, contrast, return_spectra=True
    )

    stages = sorted(frame["stage"].unique())
    sample_means = np.vstack([Y[(frame["stage"] == stage).to_numpy()].mean(axis=0) for stage in stages])
    assert pooled_relative_eigengap(spectra) == pytest.approx(audit_relgap(sample_means))


def test_pooled_ls_mean_eigengap_matches_betas_computed_independently() -> None:
    _, Y, model_full, _, ls_means, contrast = make_design()

    _, _, _, spectra = estimate_difference(
        Y, model_full, ls_means, contrast, return_spectra=True
    )

    obs_vect = np.asarray(ls_means, dtype=float) @ np.asarray(estimate_betas(model_full, Y), dtype=float)
    for index, levels in enumerate(contrast):
        expected = configuration_spectrum(obs_vect[levels, :])["relative_eigengap"]
        assert spectra["groups"][index]["relative_eigengap"] == pytest.approx(expected)


# ── 1.3 Opt-in and inertness ──────────────────────────────────────────────────


def test_estimate_difference_default_returns_the_pre_change_tuple() -> None:
    _, Y, model_full, _, ls_means, contrast = make_design()

    default = estimate_difference(Y, model_full, ls_means, contrast)
    with_spectra = estimate_difference(Y, model_full, ls_means, contrast, return_spectra=True)

    assert len(default) == 3
    assert len(with_spectra) == 4
    for baseline, extended in zip(default, with_spectra[:3]):
        np.testing.assert_array_equal(baseline, extended)


def test_rrpp_default_returns_the_pre_change_tuple() -> None:
    _, Y, model_full, model_reduced, ls_means, contrast = make_design()

    out = RRPP(
        Y, model_full, model_reduced, ls_means, contrast, permutations=7, progress=False, seed=5
    )

    assert len(out) == 3


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_rrpp_null_distributions_are_identical_with_and_without_spectra(n_jobs: int) -> None:
    """The spectrum path consumes no randomness, serial or parallel."""

    _, Y, model_full, model_reduced, ls_means, contrast = make_design()
    kwargs = dict(permutations=11, progress=False, seed=2026, n_jobs=n_jobs)

    baseline = RRPP(Y, model_full, model_reduced, ls_means, contrast, **kwargs)
    extended = RRPP(
        Y, model_full, model_reduced, ls_means, contrast, return_eigengaps=True, **kwargs
    )

    for baseline_draws, extended_draws in zip(baseline, extended[:3]):
        assert len(baseline_draws) == len(extended_draws)
        for baseline_matrix, extended_matrix in zip(baseline_draws, extended_draws):
            # Byte-identical, not merely close: the same draws produce the same
            # floating-point statistics.
            assert baseline_matrix.tobytes() == extended_matrix.tobytes()
    eigengaps = extended[3]
    assert len(eigengaps) == 11
    assert all(0.0 <= gap <= 1.0 for gap in eigengaps)


def test_rrpp_eigengaps_match_recomputing_them_from_the_permuted_fit() -> None:
    """Each recorded eigengap is the pooled one for that permutation's fit."""

    _, Y, model_full, model_reduced, ls_means, contrast = make_design()

    *_, eigengaps = RRPP(
        Y,
        model_full,
        model_reduced,
        ls_means,
        contrast,
        permutations=6,
        progress=False,
        seed=17,
        return_eigengaps=True,
    )

    # Replay the same residual randomization by hand.
    betas_reduced = estimate_betas(model_reduced, pd.DataFrame(Y))
    y_hat = np.asarray(np.matmul(model_reduced, betas_reduced), dtype=float)
    y_res = Y - y_hat
    rng = np.random.default_rng(17)
    for expected_gap in eigengaps:
        idx = rng.permutation(Y.shape[0])
        _, _, _, spectra = estimate_difference(
            y_hat + y_res[idx, :], model_full, ls_means, contrast, return_spectra=True
        )
        assert pooled_relative_eigengap(spectra) == pytest.approx(expected_gap)
