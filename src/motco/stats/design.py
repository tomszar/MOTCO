from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def center_matrix(
    dat: pd.DataFrame,
    group_col: str,
    level_col: str,
    feature_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Center feature columns by per-group means.

    Parameters
    ----------
    dat: pd.DataFrame
        Original, non-centered dataframe.
    group_col: str
        Column in `dat` indicating the group (between-subject factor).
    level_col: str
        Column in `dat` indicating the level/state (within-group factor).
    feature_cols: Sequence[str] | None
        Feature columns to center. If None, all numeric columns except
        `group_col` and `level_col` are used.

    Returns
    -------
    pd.DataFrame
        A copy of `dat` with selected feature columns centered within groups.
    """
    if feature_cols is not None:
        missing = [c for c in feature_cols if c not in dat.columns]
        if missing:
            raise ValueError(
                f"feature_cols contains column(s) not found in dat: {missing}."
            )
    datc = dat.copy()
    if feature_cols is None:
        feature_cols = [
            c
            for c in datc.select_dtypes(include=[np.number]).columns.tolist()
            if c not in {group_col, level_col}
        ]
    if not feature_cols:
        return datc
    # Center within groups using group-wise means
    datc.loc[:, feature_cols] = (
        datc.loc[:, feature_cols]
        - datc.groupby(group_col)[feature_cols].transform("mean")
    )
    return datc


def get_model_matrix(
    X: pd.DataFrame,
    group_col: str,
    level_col: str,
    full: bool = True,
) -> np.ndarray:
    """
    Build a design (model) matrix for group × level factors.

    Coding scheme
    -------------
    - Intercept (column of ones).
    - Group main effects: one-hot with drop-first for groups (G-1 columns).
    - Level main effects: one-hot with drop-first for levels (L-1 columns).
    - If `full=True`, include all interaction terms between group and level
      dummies: (G-1) × (L-1) columns.

    The category order is deterministic: sorted by string representation.

    Parameters
    ----------
    X: pd.DataFrame
        DataFrame containing `group_col` and `level_col`.
    group_col: str
        Name of the group column.
    level_col: str
        Name of the level/state column.
    full: bool
        Whether to include interaction terms.

    Returns
    -------
    np.ndarray
        Model matrix with intercept.
    """
    for col, param in [(group_col, "group_col"), (level_col, "level_col")]:
        if col not in X.columns:
            raise ValueError(
                f"{param}='{col}' not found in X. Available columns: {list(X.columns)}."
            )
        n_unique = X[col].nunique()
        if n_unique < 2:
            raise ValueError(
                f"{param}='{col}' has {n_unique} unique value(s); at least 2 are required."
            )
    # Determine deterministic category order
    g_levels = sorted(pd.unique(X[group_col].astype(str)).tolist())
    l_levels = sorted(pd.unique(X[level_col].astype(str)).tolist())
    # Wrap in Series to preserve X.index; pd.get_dummies on a bare Categorical
    # returns integer-indexed DataFrames, which causes pd.concat to outer-join with
    # the string-indexed Intercept part and double the row count.
    g = pd.Series(
        pd.Categorical(X[group_col].astype(str), categories=g_levels, ordered=True),
        index=X.index,
    )
    lc = pd.Series(
        pd.Categorical(X[level_col].astype(str), categories=l_levels, ordered=True),
        index=X.index,
    )

    G = pd.get_dummies(g, drop_first=True, dtype=int)
    L = pd.get_dummies(lc, drop_first=True, dtype=int)

    parts = []
    # Intercept
    parts.append(pd.DataFrame({"Intercept": np.ones(len(X))}, index=X.index))
    # Main effects
    if G.shape[1] > 0:
        parts.append(G)
    if L.shape[1] > 0:
        parts.append(L)
    # Interactions
    if full and G.shape[1] > 0 and L.shape[1] > 0:
        inter_cols = {}
        for g_col in G.columns:
            for l_col in L.columns:
                inter_cols[f"{g_col}:{l_col}"] = np.asarray(G[g_col]) * np.asarray(L[l_col])
        parts.append(pd.DataFrame(inter_cols, index=X.index))

    model_mat = pd.concat(parts, axis=1).to_numpy()
    return model_mat


def build_ls_means(
    group_levels: Sequence[str],
    level_levels: Sequence[str],
    full: bool = True,
) -> np.ndarray:
    """
    Generate LS-mean rows for every group × level cell consistent with
    `get_model_matrix` coding.

    Parameters
    ----------
    group_levels: Sequence[str]
        Sorted group labels; first is baseline.
    level_levels: Sequence[str]
        Sorted level labels; first is baseline.
    full: bool
        Whether to include interaction terms.

    Returns
    -------
    np.ndarray
        LS-mean design matrix with shape (G×L, 1 + (G-1) + (L-1) + I), where
        I = (G-1)×(L-1) if `full=True` else 0. Row order is by group major,
        then level minor.
    """
    g_levels = list(group_levels)
    l_levels = list(level_levels)
    Gm1 = max(len(g_levels) - 1, 0)
    Lm1 = max(len(l_levels) - 1, 0)
    n_rows = max(len(g_levels), 1) * max(len(l_levels), 1)
    n_cols = 1 + Gm1 + Lm1 + (Gm1 * Lm1 if full else 0)
    M = np.zeros((n_rows, n_cols), dtype=float)

    def row_for(i_g: int, i_l: int) -> int:
        return i_g * len(l_levels) + i_l

    # Column indices
    col = 0
    INTERCEPT = col
    col += 1
    G_START = col
    col += Gm1
    L_START = col
    col += Lm1
    I_START = col if full else None

    for gi, g_val in enumerate(g_levels):
        for li, l_val in enumerate(l_levels):
            r = row_for(gi, li)
            # Intercept
            M[r, INTERCEPT] = 1.0
            # Group dummies (drop first)
            if gi > 0 and Gm1 > 0:
                M[r, G_START + (gi - 1)] = 1.0
            # Level dummies (drop first)
            if li > 0 and Lm1 > 0:
                M[r, L_START + (li - 1)] = 1.0
            # Interactions
            if full and gi > 0 and li > 0 and (Gm1 > 0 and Lm1 > 0):
                assert I_START is not None
                idx = (gi - 1) * Lm1 + (li - 1)
                M[r, I_START + idx] = 1.0
    return M
