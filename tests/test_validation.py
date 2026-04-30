# tests/test_validation.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from motco.stats.sd import estimate_difference, RRPP, build_ls_means


def _make_simple_inputs(n_samples=10, n_features=3, n_groups=2, n_levels=2):
    """Return (Y, model_matrix, LS_means, contrast) for a 2-group 2-level design."""
    rng = np.random.default_rng(0)
    Y = rng.standard_normal((n_samples, n_features))
    # intercept + 1 group dummy + 1 level dummy + 1 interaction = 4 cols
    X = np.ones((n_samples, 4))
    X[:5, 1] = 0; X[5:, 1] = 1      # group dummy
    X[::2, 2] = 0; X[1::2, 2] = 1   # level dummy
    X[:, 3] = X[:, 1] * X[:, 2]     # interaction
    LS = build_ls_means(["A", "B"], ["t0", "t1"], full=True)  # shape (4, 4)
    contrast = [[0, 1], [2, 3]]
    return Y, X, LS, contrast


# --- estimate_difference: row mismatch ---
def test_estimate_difference_row_mismatch():
    Y, X, LS, contrast = _make_simple_inputs()
    X_bad = X[:8]  # 8 rows, Y has 10
    with pytest.raises(ValueError, match="10 rows"):
        estimate_difference(Y, X_bad, LS, contrast)


# --- estimate_difference: column mismatch ---
def test_estimate_difference_column_mismatch():
    Y, X, LS, contrast = _make_simple_inputs()
    LS_bad = np.ones((4, 3))  # 3 cols, X has 4
    with pytest.raises(ValueError, match="columns"):
        estimate_difference(Y, X, LS_bad, contrast)


# --- estimate_difference: contrast index out of bounds ---
def test_estimate_difference_contrast_oob():
    Y, X, LS, contrast = _make_simple_inputs()
    bad_contrast = [[0, 1], [2, 99]]  # index 99 is out of bounds
    with pytest.raises(ValueError, match="index 99"):
        estimate_difference(Y, X, LS, bad_contrast)


# --- estimate_difference: NaN in Y ---
def test_estimate_difference_nan_in_Y():
    Y, X, LS, contrast = _make_simple_inputs()
    Y[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        estimate_difference(Y, X, LS, contrast)


# --- estimate_difference: Inf in model_matrix ---
def test_estimate_difference_inf_in_model_matrix():
    Y, X, LS, contrast = _make_simple_inputs()
    X[0, 0] = np.inf
    with pytest.raises(ValueError, match=r"model_matrix contains NaN or Inf"):
        estimate_difference(Y, X, LS, contrast)


# --- RRPP: model_reduced row mismatch ---
def test_rrpp_reduced_row_mismatch():
    Y, X, LS, contrast = _make_simple_inputs()
    X_red = X[:8, :3]  # wrong number of rows
    with pytest.raises(ValueError, match="model_reduced"):
        RRPP(Y, X, X_red, LS, contrast, permutations=2)


# --- RRPP: NaN in model_reduced ---
def test_rrpp_nan_in_model_reduced():
    Y, X, LS, contrast = _make_simple_inputs()
    X_red = X[:, :3].copy()
    X_red[0, 0] = np.nan
    with pytest.raises(ValueError, match=r"model_reduced contains NaN"):
        RRPP(Y, X, X_red, LS, contrast, permutations=2)


from motco.stats.sd import get_model_matrix, center_matrix


# --- get_model_matrix: missing column ---
def test_get_model_matrix_missing_group_col():
    X = pd.DataFrame({"group": ["A", "B", "A", "B"], "level": ["t0", "t1", "t0", "t1"]})
    with pytest.raises(ValueError, match="'missing'"):
        get_model_matrix(X, group_col="missing", level_col="level")


def test_get_model_matrix_missing_level_col():
    X = pd.DataFrame({"group": ["A", "B", "A", "B"], "level": ["t0", "t1", "t0", "t1"]})
    with pytest.raises(ValueError, match="'missing'"):
        get_model_matrix(X, group_col="group", level_col="missing")


# --- get_model_matrix: single unique value ---
def test_get_model_matrix_single_group():
    X = pd.DataFrame({"group": ["A", "A", "A"], "level": ["t0", "t1", "t0"]})
    with pytest.raises(ValueError, match="unique"):
        get_model_matrix(X, group_col="group", level_col="level")


def test_get_model_matrix_single_level():
    X = pd.DataFrame({"group": ["A", "B", "A"], "level": ["t0", "t0", "t0"]})
    with pytest.raises(ValueError, match="unique"):
        get_model_matrix(X, group_col="group", level_col="level")


# --- center_matrix: missing feature column ---
def test_center_matrix_missing_feature_col():
    dat = pd.DataFrame({
        "group": ["A", "A", "B", "B"],
        "level": ["t0", "t1", "t0", "t1"],
        "f1": [1.0, 2.0, 3.0, 4.0],
    })
    with pytest.raises(ValueError, match="'ghost_col'"):
        center_matrix(dat, group_col="group", level_col="level", feature_cols=["ghost_col"])
