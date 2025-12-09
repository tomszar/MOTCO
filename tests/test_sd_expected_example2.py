from __future__ import annotations

import numpy as np
import pandas as pd

from motco.stats.sd import (
    get_model_matrix,
    build_ls_means,
    estimate_difference,
)


def _feature_columns(df: pd.DataFrame, group_col: str, level_col: str) -> list[str]:
    return [
        c for c in df.select_dtypes(include=[np.number]).columns if c not in {group_col, level_col}
    ]


def test_example2_expected_results_match(data_dir):
    # Fixed schema for example2
    csv_path = data_dir / "evo_649_sm_example2.csv"
    df = pd.read_csv(csv_path)
    group_col = "tax"
    level_col = "Inv"
    feat_cols = _feature_columns(df, group_col, level_col)
    assert len(feat_cols) > 0

    # Build design components
    X = df[[group_col, level_col]].copy()
    g_levels = sorted(pd.unique(X[group_col].astype(str)).tolist())
    l_levels = sorted(pd.unique(X[level_col].astype(str)).tolist())

    M = get_model_matrix(X, group_col=group_col, level_col=level_col, full=True)
    LS = build_ls_means(g_levels, l_levels, full=True)
    Y = df[feat_cols]

    # Contrast: indices per group across all levels (group-major, level-minor)
    L = len(l_levels)
    contrast: list[list[int]] = [[gi * L + li for li in range(L)] for gi in range(len(g_levels))]

    deltas, angles, shapes = estimate_difference(Y, M, LS, contrast)

    # Load ground truth
    gt_path = data_dir / "results_example2.csv"
    gt = pd.read_csv(gt_path)
    assert {"group 1", "group 2", "angle", "magnitude", "shape"}.issubset(gt.columns)

    for _, row in gt.iterrows():
        g1 = str(row["group 1"])  # e.g., "t1"
        g2 = str(row["group 2"])  # e.g., "t3"
        exp_angle = float(row["angle"])  # degrees
        exp_mag = float(row["magnitude"])  # delta magnitude difference
        exp_shape = float(row["shape"])  # shape distance

        i = g_levels.index(g1)
        j = g_levels.index(g2)

        ang = float(angles[i, j])
        mag = float(deltas[i, j])
        shp = float(shapes[i, j])

        assert np.isclose(ang, exp_angle, atol=1e-3), (
            f"Angle mismatch for {g1} vs {g2}: got {ang:.5f}, expected {exp_angle:.5f}"
        )
        assert np.isclose(mag, exp_mag, atol=1e-3), (
            f"Magnitude mismatch for {g1} vs {g2}: got {mag:.5f}, expected {exp_mag:.5f}"
        )
        assert np.isclose(shp, exp_shape, atol=1e-3), (
            f"Shape mismatch for {g1} vs {g2}: got {shp:.5f}, expected {exp_shape:.5f}"
        )
