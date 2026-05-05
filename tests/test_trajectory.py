from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from motco.stats.design import build_ls_means
from motco.stats.trajectory import estimate_difference, pair_difference


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
