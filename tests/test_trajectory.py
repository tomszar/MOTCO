from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics.pairwise import euclidean_distances

from motco.stats.design import build_ls_means
from motco.stats.trajectory import (
    _estimate_orientation,
    _estimate_shape,
    estimate_difference,
    pair_difference,
)


def _feature_columns(df: pd.DataFrame, group_col: str, level_col: str) -> list[str]:
    return [
        c for c in df.select_dtypes(include=[np.number]).columns if c not in {group_col, level_col}
    ]


def _make_simple_inputs(n_samples=10, n_features=3):
    """Return (Y, model_matrix, LS_means, contrast) for a 2-group 2-level design."""
    rng = np.random.default_rng(0)
    Y = rng.standard_normal((n_samples, n_features))
    X = np.ones((n_samples, 4))
    X[:5, 1] = 0
    X[5:, 1] = 1
    X[::2, 2] = 0
    X[1::2, 2] = 1
    X[:, 3] = X[:, 1] * X[:, 2]
    LS = build_ls_means(["A", "B"], ["t0", "t1"], full=True)
    contrast = [[0, 1], [2, 3]]
    return Y, X, LS, contrast


# ── Smoke tests ───────────────────────────────────────────────────────────────

def test_pair_difference_outputs_reasonable(example_df: pd.DataFrame, group_col: str, level_col: str):
    g_vals = sorted(pd.unique(example_df[group_col].astype(str)).tolist())
    l_vals = sorted(pd.unique(example_df[level_col].astype(str)).tolist())
    assert len(g_vals) >= 2
    assert len(l_vals) >= 2
    groups = (g_vals[0], g_vals[1])
    levels = (l_vals[0], l_vals[1])

    angle, delta = pair_difference(
        example_df,
        group_col=group_col,
        level_col=level_col,
        groups=groups,
        levels=levels,
    )

    assert isinstance(angle, float)
    assert isinstance(delta, float)
    assert 0.0 <= angle <= 180.0
    assert delta >= 0.0


# ── Validation tests ──────────────────────────────────────────────────────────

def test_estimate_difference_row_mismatch():
    Y, X, LS, contrast = _make_simple_inputs()
    X_bad = X[:8]
    with pytest.raises(ValueError, match="10 rows"):
        estimate_difference(Y, X_bad, LS, contrast)


def test_estimate_difference_column_mismatch():
    Y, X, LS, contrast = _make_simple_inputs()
    LS_bad = np.ones((4, 3))
    with pytest.raises(ValueError, match="columns"):
        estimate_difference(Y, X, LS_bad, contrast)


def test_estimate_difference_contrast_oob():
    Y, X, LS, contrast = _make_simple_inputs()
    bad_contrast = [[0, 1], [2, 99]]
    with pytest.raises(ValueError, match="index 99"):
        estimate_difference(Y, X, LS, bad_contrast)


def test_estimate_difference_nan_in_Y():
    Y, X, LS, contrast = _make_simple_inputs()
    Y[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        estimate_difference(Y, X, LS, contrast)


def test_estimate_difference_inf_in_model_matrix():
    Y, X, LS, contrast = _make_simple_inputs()
    X[0, 0] = np.inf
    with pytest.raises(ValueError, match=r"model_matrix contains NaN or Inf"):
        estimate_difference(Y, X, LS, contrast)


# ── GPA thin-QR optimization equivalence ────────────────────────────────────
#
# These reference implementations are frozen copies of the pre-optimization
# `_estimate_orientation` (eigh on the p x p covariance) and `_estimate_shape`
# (full p x p SVD in every GPA iteration, no thin-QR branch). They exist only
# to prove the optimized code in trajectory.py produces numerically identical
# output; see openspec/changes/optimize-gpa-computation/design.md.


def _estimate_orientation_reference_eigh(obs_vect: np.ndarray, levels: list[int]) -> np.ndarray:
    X = np.asarray(obs_vect, dtype=float)[levels, :]
    X = X - X.mean(axis=0, keepdims=True)
    n = X.shape[0]
    C = (X.T @ X) / (n - 1) if n > 1 else X.T @ X
    _, v = np.linalg.eigh(C)
    orientation = v[:, -1]
    c1 = float(orientation @ X[0, :])
    if c1 < 0:
        orientation = -orientation
    return orientation


def _estimate_shape_reference_full(vectors: np.ndarray, contrast: list[list[int]]) -> np.ndarray:
    V = np.asarray(vectors, dtype=float)
    n_groups = len(contrast)
    n_levels = len(contrast[0])
    n_dimensions = V.shape[1]

    X = np.empty((n_groups, n_levels, n_dimensions), dtype=float)
    for gi, levels in enumerate(contrast):
        X[gi] = V[np.asarray(levels, dtype=int), :]

    def _center_scale_unit(A: np.ndarray) -> np.ndarray:
        Z = A - A.mean(axis=0, keepdims=True)
        cs = float(np.linalg.norm(Z))
        if not np.isfinite(cs) or cs <= 1e-15:
            cs = 1.0
        return Z / cs

    temp1 = np.empty_like(X)
    for i in range(n_groups):
        temp1[i] = _center_scale_unit(X[i])

    def _pairwise_flat_dist(arr: np.ndarray) -> np.ndarray:
        flat = arr.reshape((n_groups, n_dimensions * n_levels))
        return euclidean_distances(flat)

    Qm1 = _pairwise_flat_dist(temp1)
    Qm2 = Qm1
    Q_prev_sum = float(np.tril(Qm1, k=-1).sum())
    Q_improve = Q_prev_sum

    while abs(Q_improve) > 0.00001:
        temp2 = np.empty_like(temp1)
        for i in range(n_groups):
            M = temp1[np.arange(n_groups) != i].mean(axis=0) if n_groups > 1 else temp1[i]

            Z1 = _center_scale_unit(temp1[i])
            Z2 = _center_scale_unit(M)

            H = Z2.T @ Z1
            U, S, Vt = np.linalg.svd(H, full_matrices=False)
            Vv = Vt.T

            detH = float(np.linalg.det(H))
            sig = -1.0 if detH < 0.0 else 1.0

            U[:, -1] *= sig
            Gam = Vv @ U.T

            if S.size == 0:
                beta = 0.0
            elif S.size == 1:
                beta = float(sig * S[0])
            else:
                beta = float(np.sum(S[:-1]) + sig * S[-1])

            temp2[i] = beta * (Z1 @ Gam)

        Qm2 = _pairwise_flat_dist(temp2)
        Q_sum = float(np.tril(Qm2, k=-1).sum())
        Q_improve = Q_prev_sum - Q_sum
        Q_prev_sum = Q_sum
        temp1 = temp2

    return Qm2


@pytest.mark.parametrize("n_levels,n_dimensions", [(4, 658), (4, 10), (4, 3)])
def test_gpa_thin_matches_full(n_levels, n_dimensions):
    rng = np.random.default_rng(42)
    observed_vectors = rng.standard_normal((2 * n_levels, n_dimensions))
    contrast = [list(range(n_levels)), list(range(n_levels, 2 * n_levels))]

    reference = _estimate_shape_reference_full(observed_vectors, contrast)
    result = _estimate_shape(observed_vectors, contrast)

    np.testing.assert_allclose(result, reference, atol=1e-10)


@pytest.mark.parametrize("n_levels,n_dimensions", [(4, 658), (4, 10), (4, 3)])
def test_orientation_thin_matches_full(n_levels, n_dimensions):
    rng = np.random.default_rng(7)
    observed_vectors = rng.standard_normal((n_levels, n_dimensions))
    levels = list(range(n_levels))

    reference = _estimate_orientation_reference_eigh(observed_vectors, levels)
    result = _estimate_orientation(observed_vectors, levels)

    np.testing.assert_allclose(result, reference, atol=1e-10)
