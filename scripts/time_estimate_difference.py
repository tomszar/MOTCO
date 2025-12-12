from __future__ import annotations

import argparse
import statistics as stats
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import sys
_ROOT = Path(__file__).resolve().parents[1]
# Ensure local src/ is importable when running the script directly
sys.path.insert(0, str(_ROOT / "src"))

from motco.stats.sd import (
    get_model_matrix,
    build_ls_means,
    estimate_betas,
    _estimate_orientation,
    _estimate_size,
    _estimate_shape,
)


def _feature_columns(df: pd.DataFrame, group_col: str, level_col: str) -> list[str]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in num_cols if c not in {group_col, level_col}]


def _default_data_path() -> Path:
    # Project root is parent of this file's directory (scripts/)
    root = Path(__file__).resolve().parents[1]
    return root / "tests" / "data" / "evo_649_sm_example2.csv"


def _build_design(df: pd.DataFrame, group_col: str, level_col: str):
    X = df[[group_col, level_col]].copy()
    g_levels = sorted(pd.unique(X[group_col].astype(str)).tolist())
    l_levels = sorted(pd.unique(X[level_col].astype(str)).tolist())

    M_full = get_model_matrix(X, group_col=group_col, level_col=level_col, full=True)
    LS = build_ls_means(g_levels, l_levels, full=True)

    # Contrast: indices per group across all levels (group-major, level-minor)
    L = len(l_levels)
    contrast: list[list[int]] = [[gi * L + li for li in range(L)] for gi in range(len(g_levels))]

    return M_full, LS, contrast


def _time_functions(
    Y: pd.DataFrame | np.ndarray,
    M_full: pd.DataFrame | np.ndarray,
    LS: pd.DataFrame | np.ndarray,
    contrast: list[list[int]],
    repeats: int,
):
    t_betas: list[float] = []
    t_orient: list[float] = []
    t_size: list[float] = []
    t_shape: list[float] = []

    n_groups = len(contrast)

    for r in range(repeats):
        # estimate_betas
        s = time.perf_counter()
        betas = estimate_betas(M_full, Y)
        t_betas.append(time.perf_counter() - s)

        # Build observed vectors once per iteration (LS means × betas)
        obs_vect = pd.DataFrame(np.matmul(np.asarray(LS, dtype=float), np.asarray(betas, dtype=float)))

        # _estimate_orientation for all groups
        s = time.perf_counter()
        for i in range(n_groups):
            _ = _estimate_orientation(obs_vect, contrast[i])
        t_orient.append(time.perf_counter() - s)

        # _estimate_size for all groups
        s = time.perf_counter()
        for i in range(n_groups):
            _ = _estimate_size(obs_vect, contrast[i])
        t_size.append(time.perf_counter() - s)

        # _estimate_shape once
        s = time.perf_counter()
        _ = _estimate_shape(obs_vect, contrast)
        t_shape.append(time.perf_counter() - s)

    return t_betas, t_orient, t_size, t_shape


def _summarize(name: str, values: Sequence[float]) -> str:
    if not values:
        return f"{name}: no data"
    mean = stats.fmean(values)
    stdev = stats.pstdev(values) if len(values) > 1 else 0.0
    mn = min(values)
    mx = max(values)
    return (
        f"{name:>20}: mean={mean*1000:.3f} ms  std={stdev*1000:.3f} ms  "
        f"min={mn*1000:.3f} ms  max={mx*1000:.3f} ms"
    )


def main():
    p = argparse.ArgumentParser(
        description=(
            "Time components of the estimate_difference workflow on example2 data: "
            "estimate_betas, _estimate_orientation (all groups), _estimate_size (all groups), and _estimate_shape."
        )
    )
    p.add_argument(
        "--data",
        type=Path,
        default=_default_data_path(),
        help="Path to evo_649_sm_example2.csv (defaults to tests/data/evo_649_sm_example2.csv)",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Number of repetitions (>= 10 recommended)",
    )
    args = p.parse_args()

    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")

    # Fixed schema for example2
    group_col = "tax"
    level_col = "Inv"

    df = pd.read_csv(args.data)
    feat_cols = _feature_columns(df, group_col, level_col)
    if not feat_cols:
        raise SystemExit("No numeric feature columns found in the dataset.")

    # Use the first 2 principal components of the feature matrix as the response Y
    pca = PCA(n_components=2)
    Y = pd.DataFrame(pca.fit_transform(df[feat_cols]))
    M_full, LS, contrast = _build_design(df, group_col, level_col)

    print(f"Data: {args.data}")
    print(f"Rows: {len(df)}, Features: {len(feat_cols)}; Groups: {len(contrast)}")
    print(f"Repeats: {args.repeats}")

    t_betas, t_orient, t_size, t_shape = _time_functions(Y, M_full, LS, contrast, args.repeats)

    # Per-iteration report
    print("\nPer-iteration times (ms):")
    for i in range(args.repeats):
        print(
            f"  Iter {i+1:2d}: betas={t_betas[i]*1000:.3f}  "
            f"orient(all)={t_orient[i]*1000:.3f}  size(all)={t_size[i]*1000:.3f}  shape={t_shape[i]*1000:.3f}"
        )

    # Summary
    print("\nSummary:")
    print(_summarize("estimate_betas", t_betas))
    print(_summarize("_estimate_orientation", t_orient))
    print(_summarize("_estimate_size", t_size))
    print(_summarize("_estimate_shape", t_shape))


if __name__ == "__main__":
    main()
