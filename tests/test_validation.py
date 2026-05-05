# tests/test_validation.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from motco.stats.pls import plsda_doubleCV
from motco.stats.snf import SNF, get_affinity_matrix


def _small_pls_args():
    """Minimal valid args for plsda_doubleCV (fast: 2-fold, 1 repeat, 2 components)."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((20, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(["A"] * 10 + ["B"] * 10)
    return X, y


# --- plsda_doubleCV: row mismatch ---
def test_plsda_row_mismatch():
    X, y = _small_pls_args()
    y_bad = y.iloc[:18]
    with pytest.raises(ValueError, match="20 rows"):
        plsda_doubleCV(X, y_bad, cv1_splits=2, cv2_splits=2, n_repeats=1, max_components=2)


# --- plsda_doubleCV: single class ---
def test_plsda_single_class():
    X, _ = _small_pls_args()
    y_one = pd.Series(["A"] * 20)
    with pytest.raises(ValueError, match="class"):
        plsda_doubleCV(X, y_one, cv1_splits=2, cv2_splits=2, n_repeats=1, max_components=2)


# --- plsda_doubleCV: max_components too large ---
def test_plsda_max_components_exceeds_features():
    X, y = _small_pls_args()
    with pytest.raises(ValueError, match="max_components"):
        plsda_doubleCV(X, y, cv1_splits=2, cv2_splits=2, n_repeats=1, max_components=100)


# --- plsda_doubleCV: NaN in X ---
def test_plsda_nan_in_X():
    X, y = _small_pls_args()
    X.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        plsda_doubleCV(X, y, cv1_splits=2, cv2_splits=2, n_repeats=1, max_components=2)


def _square_affinity(n=10):
    rng = np.random.default_rng(0)
    W = rng.uniform(0, 1, (n, n))
    return (W + W.T) / 2


# --- get_affinity_matrix: misaligned datasets ---
def test_affinity_matrix_row_mismatch():
    rng = np.random.default_rng(0)
    d1 = rng.standard_normal((10, 5))
    d2 = rng.standard_normal((8, 5))
    with pytest.raises(ValueError, match="8 rows"):
        get_affinity_matrix([d1, d2], K=3)


# --- get_affinity_matrix: K >= n_samples ---
def test_affinity_matrix_K_too_large():
    rng = np.random.default_rng(0)
    d1 = rng.standard_normal((5, 3))
    d2 = rng.standard_normal((5, 3))
    with pytest.raises(ValueError, match="K=10"):
        get_affinity_matrix([d1, d2], K=10)


# --- get_affinity_matrix: eps <= 0 ---
def test_affinity_matrix_invalid_eps():
    rng = np.random.default_rng(0)
    d1 = rng.standard_normal((10, 3))
    d2 = rng.standard_normal((10, 3))
    with pytest.raises(ValueError, match="eps"):
        get_affinity_matrix([d1, d2], K=3, eps=-0.1)


# --- get_affinity_matrix: NaN in data ---
def test_affinity_matrix_nan():
    rng = np.random.default_rng(0)
    d1 = rng.standard_normal((10, 3))
    d2 = rng.standard_normal((10, 3))
    d2[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        get_affinity_matrix([d1, d2], K=3)


# --- SNF: non-square matrix ---
def test_snf_non_square():
    W1 = _square_affinity(5)
    W2 = np.ones((5, 6))
    with pytest.raises(ValueError, match="not square"):
        SNF([W1, W2])


# --- SNF: shape mismatch ---
def test_snf_shape_mismatch():
    W1 = _square_affinity(5)
    W2 = _square_affinity(6)
    with pytest.raises(ValueError, match="shape"):
        SNF([W1, W2])


# --- SNF: k >= n_samples ---
def test_snf_k_too_large():
    W1 = _square_affinity(5)
    W2 = _square_affinity(5)
    with pytest.raises(ValueError, match="k=10"):
        SNF([W1, W2], k=10)
