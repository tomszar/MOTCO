from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

from motco.stats.design import build_ls_means, get_model_matrix
from motco.stats.permutation import RRPP
from motco.stats.trajectory import estimate_difference, get_observed_vectors


def _feature_columns(df: pd.DataFrame, group_col: str, level_col: str) -> list[str]:
    return [
        c for c in df.select_dtypes(include=[np.number]).columns if c not in {group_col, level_col}
    ]


PERMS = int(os.getenv("MOTCO_TEST_PERMS", "10000"))
N_JOBS = int(os.getenv("MOTCO_N_JOBS", "1"))


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


def _make_three_stage_shape_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[int]]]:
    base = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.25],
            [1.75, 1.5],
            [0.25, 2.25],
        ],
        dtype=float,
    )
    bend = base.copy()
    bend[1] += np.array([0.45, -0.65])
    Y = np.vstack([base, bend])
    model_full = np.eye(Y.shape[0])
    model_reduced = np.ones((Y.shape[0], 1))
    ls_means = np.eye(Y.shape[0])
    contrast = [list(range(base.shape[0])), list(range(base.shape[0], 2 * base.shape[0]))]
    return Y, model_full, model_reduced, ls_means, contrast


# ── Validation tests ──────────────────────────────────────────────────────────

def test_rrpp_reduced_row_mismatch():
    Y, X, LS, contrast = _make_simple_inputs()
    X_red = X[:8, :3]
    with pytest.raises(ValueError, match="model_reduced"):
        RRPP(Y, X, X_red, LS, contrast, permutations=2)


def test_rrpp_nan_in_model_reduced():
    Y, X, LS, contrast = _make_simple_inputs()
    X_red = X[:, :3].copy()
    X_red[0, 0] = np.nan
    with pytest.raises(ValueError, match=r"model_reduced contains NaN"):
        RRPP(Y, X, X_red, LS, contrast, permutations=2)


def test_rrpp_returns_shape_matrices_for_three_stage_contrast():
    Y, model_full, model_reduced, ls_means, contrast = _make_three_stage_shape_inputs()

    _, _, dist_shape = RRPP(
        Y,
        model_full,
        model_reduced,
        ls_means,
        contrast,
        permutations=3,
        progress=False,
        seed=42,
    )

    assert len(dist_shape) == 3
    for shape_matrix in dist_shape:
        assert shape_matrix.shape == (2, 2)
        np.testing.assert_allclose(shape_matrix, shape_matrix.T, atol=1e-10)


# ── Regression tests ──────────────────────────────────────────────────────────

@pytest.mark.slow
def test_example1_expected_results_match(data_dir):
    csv_path = data_dir / "evo_649_sm_example1.csv"
    df = pd.read_csv(csv_path)
    group_col = "taxa"
    level_col = "Inv"
    feat_cols = _feature_columns(df, group_col, level_col)

    X = df[[group_col, level_col]].copy()
    g_levels = sorted(pd.unique(X[group_col].astype(str)).tolist())
    l_levels = sorted(pd.unique(X[level_col].astype(str)).tolist())

    M_full = get_model_matrix(X, group_col=group_col, level_col=level_col, full=True)
    LS = build_ls_means(g_levels, l_levels, full=True)
    pca = PCA(n_components=2)
    Y = pd.DataFrame(pca.fit_transform(df[feat_cols]))

    M_red = get_model_matrix(X, group_col=group_col, level_col=level_col, full=False)

    L = len(l_levels)
    contrast: list[list[int]] = [[gi * L + li for li in range(L)] for gi in range(len(g_levels))]

    deltas, angles, _ = estimate_difference(Y, M_full, LS, contrast)

    # Expected angles under the progression convention: with two stages the
    # pairwise angle is identically the angle between the unit transition
    # vectors of the same fitted LS means
    # (openspec/specs/trajectory-orientation-invariance).
    assert L == 2, "example1 is the two-stage fixture"
    obs_vect = np.asarray(
        get_observed_vectors(X, Y, group_col=group_col, level_col=level_col, full=True),
        dtype=float,
    )
    transitions = [obs_vect[2 * gi + 1] - obs_vect[2 * gi] for gi in range(len(g_levels))]
    units = [t / np.linalg.norm(t) for t in transitions]

    dist_delta, dist_angle, _ = RRPP(
        Y, M_full, M_red, LS, contrast, permutations=PERMS, n_jobs=N_JOBS
    )

    gt_path = data_dir / "results_example1.csv"
    gt = pd.read_csv(gt_path)
    assert {"group 1", "group 2", "angle", "magnitude", "angle_pvalue", "magnitude_pvalue"}.issubset(
        gt.columns
    )

    def _pval_right_tailed(samples: list[np.ndarray], obs: float, i: int, j: int) -> float:
        vals = np.array([s[i, j] for s in samples], dtype=float)
        return (float(np.sum(vals >= obs)) + 1.0) / (len(vals) + 1.0)

    for _, row in gt.iterrows():
        g1 = str(row["group 1"])
        g2 = str(row["group 2"])
        exp_angle = float(row["angle"])
        exp_mag = float(row["magnitude"])
        exp_angle_p = float(row["angle_pvalue"])
        exp_mag_p = float(row["magnitude_pvalue"])

        i = g_levels.index(g1)
        j = g_levels.index(g2)
        exp_direct = float(np.degrees(np.arccos(np.clip(units[i] @ units[j], -1.0, 1.0))))

        ang = float(angles[i, j])
        mag = float(deltas[i, j])
        p_ang = _pval_right_tailed(dist_angle, ang, i, j)
        p_mag = _pval_right_tailed(dist_delta, mag, i, j)

        print(f"\nComparing {g1} vs {g2}:")
        print(f"  Angle:     {ang:10.5f} (expected: {exp_direct:10.5f}; committed: {exp_angle:10.5f})")
        print(f"  Magnitude: {mag:10.5f} (expected: {exp_mag:10.5f})")
        print(f"  Angle p:   {p_ang:10.4f} (expected: {exp_angle_p:10.4f})")
        print(f"  Mag   p:   {p_mag:10.4f} (expected: {exp_mag_p:10.4f})")

        # The committed R angle is the direct-vector angle up to a supplement
        # (raw sign anchor, evo_649_sm_suppmat.r:64). The expectation itself is
        # the direct-vector angle — a supplement in the output must fail.
        assert min(abs(exp_angle - exp_direct), abs(exp_angle - (180.0 - exp_direct))) < 1e-3, (
            f"Committed angle for {g1} vs {g2} is neither the direct-vector angle nor"
            f" its supplement: committed {exp_angle:.5f}, direct {exp_direct:.5f}"
        )
        assert np.isclose(ang, exp_direct, atol=1e-6), (
            f"Angle mismatch for {g1} vs {g2}: got {ang:.5f}, expected {exp_direct:.5f}"
        )
        assert np.isclose(mag, exp_mag, atol=1e-3), (
            f"Magnitude mismatch for {g1} vs {g2}: got {mag:.5f}, expected {exp_mag:.5f}"
        )

        alpha = 0.05
        exp_ang_sig = exp_angle_p < alpha
        exp_mag_sig = exp_mag_p < alpha
        est_ang_sig = p_ang < alpha
        est_mag_sig = p_mag < alpha

        print(
            f"  Angle sig: {'SIG' if est_ang_sig else 'NS '} (expected: {'SIG' if exp_ang_sig else 'NS '})"
        )
        print(
            f"  Mag   sig: {'SIG' if est_mag_sig else 'NS '} (expected: {'SIG' if exp_mag_sig else 'NS '})"
        )

        assert est_ang_sig == exp_ang_sig, (
            f"Angle significance mismatch for {g1} vs {g2}: got {'SIG' if est_ang_sig else 'NS'},"
            f" expected {'SIG' if exp_ang_sig else 'NS'} (p_est={p_ang:.4f}, p_exp={exp_angle_p:.4f})"
        )
        assert est_mag_sig == exp_mag_sig, (
            f"Magnitude significance mismatch for {g1} vs {g2}: got {'SIG' if est_mag_sig else 'NS'},"
            f" expected {'SIG' if exp_mag_sig else 'NS'} (p_est={p_mag:.4f}, p_exp={exp_mag_p:.4f})"
        )


@pytest.mark.slow
def test_example2_expected_results_match(data_dir):
    csv_path = data_dir / "evo_649_sm_example2.csv"
    df = pd.read_csv(csv_path)
    group_col = "tax"
    level_col = "Inv"
    feat_cols = _feature_columns(df, group_col, level_col)
    assert len(feat_cols) > 0

    X = df[[group_col, level_col]].copy()
    g_levels = sorted(pd.unique(X[group_col].astype(str)).tolist())
    l_levels = sorted(pd.unique(X[level_col].astype(str)).tolist())

    M_full = get_model_matrix(X, group_col=group_col, level_col=level_col, full=True)
    LS = build_ls_means(g_levels, l_levels, full=True)
    pca = PCA(n_components=2)
    Y = pd.DataFrame(pca.fit_transform(df[feat_cols]))

    M_red = get_model_matrix(X, group_col=group_col, level_col=level_col, full=False)

    L = len(l_levels)
    contrast: list[list[int]] = [[gi * L + li for li in range(L)] for gi in range(len(g_levels))]

    deltas, angles, shapes = estimate_difference(Y, M_full, LS, contrast)

    dist_delta, dist_angle, dist_shape = RRPP(
        Y, M_full, M_red, LS, contrast, permutations=PERMS, n_jobs=N_JOBS
    )

    gt_path = data_dir / "results_example2.csv"
    gt = pd.read_csv(gt_path)
    assert {
        "group 1", "group 2", "angle", "magnitude", "angle_pvalue", "magnitude_pvalue",
    }.issubset(gt.columns)

    def _pval_right_tailed(samples: list[np.ndarray], obs: float, i: int, j: int) -> float:
        vals = np.array([s[i, j] for s in samples], dtype=float)
        return (float(np.sum(vals >= obs)) + 1.0) / (len(vals) + 1.0)

    for _, row in gt.iterrows():
        g1 = str(row["group 1"])
        g2 = str(row["group 2"])
        exp_angle = float(row["angle"])
        exp_mag = float(row["magnitude"])
        exp_angle_p = float(row["angle_pvalue"])
        exp_mag_p = float(row["magnitude_pvalue"])

        i = g_levels.index(g1)
        j = g_levels.index(g2)

        ang = float(angles[i, j])
        mag = float(deltas[i, j])
        shp = float(shapes[i, j])

        p_ang = _pval_right_tailed(dist_angle, ang, i, j)
        p_mag = _pval_right_tailed(dist_delta, mag, i, j)
        p_shp = _pval_right_tailed(dist_shape, shp, i, j)

        print(f"\nComparing {g1} vs {g2}:")
        print(f"  Angle:     {ang:10.5f} (expected: {exp_angle:10.5f} or {180.0 - exp_angle:10.5f})")
        print(f"  Magnitude: {mag:10.5f} (expected: {exp_mag:10.5f})")
        print(f"  Shape:     {shp:10.5f} (legacy CSV shape values are superseded)")
        print(f"  Angle p:   {p_ang:10.4f} (expected: {exp_angle_p:10.4f})")
        print(f"  Mag   p:   {p_mag:10.4f} (expected: {exp_mag_p:10.4f})")
        print(f"  Shape p:   {p_shp:10.4f} (legacy CSV shape p-values are superseded)")

        angle_ok = np.isclose(ang, exp_angle, atol=1e-1) or np.isclose(
            ang, 180.0 - exp_angle, atol=1e-1
        )
        assert angle_ok, (
            f"Angle mismatch for {g1} vs {g2}: got {ang:.5f}, expected {exp_angle:.5f}"
            f" (accepting 180-exp as well: {(180.0 - exp_angle):.5f})"
        )
        assert np.isclose(mag, exp_mag, atol=1e-1), (
            f"Magnitude mismatch for {g1} vs {g2}: got {mag:.5f}, expected {exp_mag:.5f}"
        )
        assert shp >= 0.0

        alpha = 0.05
        est_ang_sig = p_ang < alpha
        est_mag_sig = p_mag < alpha

        print(
            f"  Angle sig: {'SIG' if est_ang_sig else 'NS '} (expected: {'SIG' if exp_angle_p < alpha else 'NS '})"
        )
        print(
            f"  Mag   sig: {'SIG' if est_mag_sig else 'NS '} (expected: {'SIG' if exp_mag_p < alpha else 'NS '})"
        )
        print(f"  Shape sig: {'SIG' if p_shp < alpha else 'NS '} (legacy CSV shape significance is superseded)")

        assert est_ang_sig == (exp_angle_p < alpha), (
            f"Angle significance mismatch for {g1} vs {g2}: got {'SIG' if est_ang_sig else 'NS'},"
            f" expected {'SIG' if exp_angle_p < alpha else 'NS'} (p_est={p_ang:.4f}, p_exp={exp_angle_p:.4f})"
        )
        assert est_mag_sig == (exp_mag_p < alpha), (
            f"Magnitude significance mismatch for {g1} vs {g2}: got {'SIG' if est_mag_sig else 'NS'},"
            f" expected {'SIG' if exp_mag_p < alpha else 'NS'} (p_est={p_mag:.4f}, p_exp={exp_mag_p:.4f})"
        )
