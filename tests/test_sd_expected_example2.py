from __future__ import annotations

import os
import numpy as np
import pandas as pd

from motco.stats.sd import (
    get_model_matrix,
    build_ls_means,
    estimate_difference,
    RRPP,
)
from sklearn.decomposition import PCA


def _feature_columns(df: pd.DataFrame, group_col: str, level_col: str) -> list[str]:
    return [
        c for c in df.select_dtypes(include=[np.number]).columns if c not in {group_col, level_col}
    ]

PERMS = int(os.getenv("MOTCO_TEST_PERMS", "10000"))


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

    M_full = get_model_matrix(X, group_col=group_col, level_col=level_col, full=True)
    LS = build_ls_means(g_levels, l_levels, full=True)
    # Use the first 2 principal components of the feature matrix as the response Y
    pca = PCA(n_components=2)
    Y = pd.DataFrame(pca.fit_transform(df[feat_cols]))

    # Reduced model: collapse groups → only intercept + level effects
    X_red = X.copy()
    X_red.loc[:, group_col] = g_levels[0]
    M_red = get_model_matrix(X_red, group_col=group_col, level_col=level_col, full=True)

    # Contrast: indices per group across all levels (group-major, level-minor)
    L = len(l_levels)
    contrast: list[list[int]] = [[gi * L + li for li in range(L)] for gi in range(len(g_levels))]

    deltas, angles, shapes = estimate_difference(Y, M_full, LS, contrast)

    # RRPP distributions using configurable permutations (default 1,000)
    dist_delta, dist_angle, dist_shape = RRPP(
        Y, M_full, M_red, LS, contrast, permutations=PERMS, n_jobs=-1
    )

    # Load ground truth
    gt_path = data_dir / "results_example2.csv"
    gt = pd.read_csv(gt_path)
    assert {
        "group 1",
        "group 2",
        "angle",
        "magnitude",
        "shape",
        "angle_pvalue",
        "magnitude_pvalue",
        "shape_pvalue",
    }.issubset(gt.columns)

    def _pval_right_tailed(samples: list[np.ndarray], obs: float, i: int, j: int) -> float:
        vals = np.array([s[i, j] for s in samples], dtype=float)
        return (float(np.sum(vals >= obs)) + 1.0) / (len(vals) + 1.0)

    for _, row in gt.iterrows():
        g1 = str(row["group 1"])  # e.g., "t1"
        g2 = str(row["group 2"])  # e.g., "t3"
        exp_angle = float(row["angle"])  # degrees
        exp_mag = float(row["magnitude"])  # delta magnitude difference
        exp_shape = float(row["shape"])  # shape distance
        exp_angle_p = float(row["angle_pvalue"])  # permutation p-value
        exp_mag_p = float(row["magnitude_pvalue"])  # permutation p-value
        exp_shape_p = float(row["shape_pvalue"])  # permutation p-value

        i = g_levels.index(g1)
        j = g_levels.index(g2)

        ang = float(angles[i, j])
        mag = float(deltas[i, j])
        shp = float(shapes[i, j])

        p_ang = _pval_right_tailed(dist_angle, ang, i, j)
        p_mag = _pval_right_tailed(dist_delta, mag, i, j)
        p_shp = _pval_right_tailed(dist_shape, shp, i, j)

        print(f"\nComparing {g1} vs {g2}:")
        print(f"  Angle:     {ang:10.5f} (expected: {exp_angle:10.5f})")
        print(f"  Magnitude: {mag:10.5f} (expected: {exp_mag:10.5f})")
        print(f"  Shape:     {shp:10.5f} (expected: {exp_shape:10.5f})")
        print(f"  Angle p:   {p_ang:10.4f} (expected: {exp_angle_p:10.4f})")
        print(f"  Mag   p:   {p_mag:10.4f} (expected: {exp_mag_p:10.4f})")
        print(f"  Shape p:   {p_shp:10.4f} (expected: {exp_shape_p:10.4f})")

        assert np.isclose(ang, exp_angle, atol=1e-1), (
            f"Angle mismatch for {g1} vs {g2}: got {ang:.5f}, expected {exp_angle:.5f}"
        )
        assert np.isclose(mag, exp_mag, atol=1e-1), (
            f"Magnitude mismatch for {g1} vs {g2}: got {mag:.5f}, expected {exp_mag:.5f}"
        )
        assert np.isclose(shp, exp_shape, atol=1e-1), (
            f"Shape mismatch for {g1} vs {g2}: got {shp:.5f}, expected {exp_shape:.5f}"
        )

        # P-value significance comparisons at alpha = 0.05
        alpha = 0.05
        exp_ang_sig = exp_angle_p < alpha
        exp_mag_sig = exp_mag_p < alpha
        exp_shp_sig = exp_shape_p < alpha
        est_ang_sig = p_ang < alpha
        est_mag_sig = p_mag < alpha
        est_shp_sig = p_shp < alpha

        print(
            f"  Angle sig: {'SIG' if est_ang_sig else 'NS '} (expected: {'SIG' if exp_ang_sig else 'NS '})"
        )
        print(
            f"  Mag   sig: {'SIG' if est_mag_sig else 'NS '} (expected: {'SIG' if exp_mag_sig else 'NS '})"
        )
        print(
            f"  Shape sig: {'SIG' if est_shp_sig else 'NS '} (expected: {'SIG' if exp_shp_sig else 'NS '})"
        )

        assert est_ang_sig == exp_ang_sig, (
            f"Angle significance mismatch for {g1} vs {g2}: got {'SIG' if est_ang_sig else 'NS'},"
            f" expected {'SIG' if exp_ang_sig else 'NS'} (p_est={p_ang:.4f}, p_exp={exp_angle_p:.4f})"
        )
        assert est_mag_sig == exp_mag_sig, (
            f"Magnitude significance mismatch for {g1} vs {g2}: got {'SIG' if est_mag_sig else 'NS'},"
            f" expected {'SIG' if exp_mag_sig else 'NS'} (p_est={p_mag:.4f}, p_exp={exp_mag_p:.4f})"
        )
        assert est_shp_sig == exp_shp_sig, (
            f"Shape significance mismatch for {g1} vs {g2}: got {'SIG' if est_shp_sig else 'NS'},"
            f" expected {'SIG' if exp_shp_sig else 'NS'} (p_est={p_shp:.4f}, p_exp={exp_shape_p:.4f})"
        )
