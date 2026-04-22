from __future__ import annotations

import numpy as np
import pandas as pd

from motco.stats.sd import (
    build_ls_means,
    center_matrix,
    get_model_matrix,
    pair_difference,
)


def _feature_columns(df: pd.DataFrame, group_col: str, level_col: str) -> list[str]:
    return [
        c for c in df.select_dtypes(include=[np.number]).columns if c not in {group_col, level_col}
    ]


def test_center_matrix_group_means_zero(example_df: pd.DataFrame, group_col: str, level_col: str):
    feat_cols = _feature_columns(example_df, group_col, level_col)
    centered = center_matrix(example_df, group_col=group_col, level_col=level_col, feature_cols=feat_cols)

    # Within each group, feature means after centering should be ~0
    grp_means = centered.groupby(group_col)[feat_cols].mean()
    assert np.allclose(grp_means.values, 0.0, atol=1e-10)


def test_model_matrix_and_ls_means_shapes(example_df: pd.DataFrame, group_col: str, level_col: str):
    X = example_df[[group_col, level_col]].copy()
    g_levels = sorted(pd.unique(X[group_col].astype(str)).tolist())
    l_levels = sorted(pd.unique(X[level_col].astype(str)).tolist())

    M = get_model_matrix(X, group_col=group_col, level_col=level_col, full=True)
    # Expected columns: 1 + (G-1) + (L-1) + (G-1)*(L-1)
    Gm1 = max(len(g_levels) - 1, 0)
    Lm1 = max(len(l_levels) - 1, 0)
    expected_cols = 1 + Gm1 + Lm1 + (Gm1 * Lm1)
    assert M.shape[0] == len(X)
    assert M.shape[1] == expected_cols

    LS = build_ls_means(g_levels, l_levels, full=True)
    assert LS.shape[0] == len(g_levels) * len(l_levels)
    assert LS.shape[1] == expected_cols


def test_pair_difference_outputs_reasonable(example_df: pd.DataFrame, group_col: str, level_col: str):
    # pick first two groups and first two levels to satisfy function requirements
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
