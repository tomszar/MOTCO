from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from motco.stats.design import build_ls_means, center_matrix, get_model_matrix


def _feature_columns(df: pd.DataFrame, group_col: str, level_col: str) -> list[str]:
    return [
        c for c in df.select_dtypes(include=[np.number]).columns if c not in {group_col, level_col}
    ]


# ── Smoke tests ───────────────────────────────────────────────────────────────

def test_center_matrix_group_means_zero(example_df: pd.DataFrame, group_col: str, level_col: str):
    feat_cols = _feature_columns(example_df, group_col, level_col)
    centered = center_matrix(example_df, group_col=group_col, level_col=level_col, feature_cols=feat_cols)

    grp_means = centered.groupby(group_col)[feat_cols].mean()
    assert np.allclose(grp_means.values, 0.0, atol=1e-10)


def test_model_matrix_and_ls_means_shapes(example_df: pd.DataFrame, group_col: str, level_col: str):
    X = example_df[[group_col, level_col]].copy()
    g_levels = sorted(pd.unique(X[group_col].astype(str)).tolist())
    l_levels = sorted(pd.unique(X[level_col].astype(str)).tolist())

    M = get_model_matrix(X, group_col=group_col, level_col=level_col, full=True)
    Gm1 = max(len(g_levels) - 1, 0)
    Lm1 = max(len(l_levels) - 1, 0)
    expected_cols = 1 + Gm1 + Lm1 + (Gm1 * Lm1)
    assert M.shape[0] == len(X)
    assert M.shape[1] == expected_cols

    LS = build_ls_means(g_levels, l_levels, full=True)
    assert LS.shape[0] == len(g_levels) * len(l_levels)
    assert LS.shape[1] == expected_cols


# ── Validation tests ──────────────────────────────────────────────────────────

def test_get_model_matrix_missing_group_col():
    X = pd.DataFrame({"group": ["A", "B", "A", "B"], "level": ["t0", "t1", "t0", "t1"]})
    with pytest.raises(ValueError, match="'missing'"):
        get_model_matrix(X, group_col="missing", level_col="level")


def test_get_model_matrix_missing_level_col():
    X = pd.DataFrame({"group": ["A", "B", "A", "B"], "level": ["t0", "t1", "t0", "t1"]})
    with pytest.raises(ValueError, match="'missing'"):
        get_model_matrix(X, group_col="group", level_col="missing")


def test_get_model_matrix_single_group():
    X = pd.DataFrame({"group": ["A", "A", "A"], "level": ["t0", "t1", "t0"]})
    with pytest.raises(ValueError, match="unique"):
        get_model_matrix(X, group_col="group", level_col="level")


def test_get_model_matrix_single_level():
    X = pd.DataFrame({"group": ["A", "B", "A"], "level": ["t0", "t0", "t0"]})
    with pytest.raises(ValueError, match="unique"):
        get_model_matrix(X, group_col="group", level_col="level")


def test_center_matrix_missing_feature_col():
    dat = pd.DataFrame({
        "group": ["A", "A", "B", "B"],
        "level": ["t0", "t1", "t0", "t1"],
        "f1": [1.0, 2.0, 3.0, 4.0],
    })
    with pytest.raises(ValueError, match="'ghost_col'"):
        center_matrix(dat, group_col="group", level_col="level", feature_cols=["ghost_col"])
