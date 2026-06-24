"""Rung-4 cross-omic coupling probe driver.

Runs all analysis axes defined in
``openspec/changes/rung4-cross-omic-coupling/design.md`` and prints the
headline tables to stdout.  Results inform the findings writeup.

Usage::

    .venv/bin/python scripts/coupling_recovery_probe.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd

from motco.simulations.coupling_recovery import (
    CouplingRecoveryParams,
    plot_analytic_comparison,
    plot_coupling_sweep,
    run_analytic_comparison,
    run_coupling_sweep,
    run_dim_ratio_sweep,
    run_matrix_seed_sweep,
)

pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 120)

SEEDS = list(range(10))
BASE = CouplingRecoveryParams(
    seed=0,
    n_features_anchor=50,
    n_samples_per_cell=40,
    noise_scale=1.0,
    signal_scale=5.0,
    dim_ratio=1.0,
    coupling_scale=0.5,
    m_structure="random_sparse",
    nnz_per_nuis=3,
    matrix_seed=0,
    n_components=10,
)


def _header(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}\n")


def axis1_coupling_sweep() -> None:
    _header("Axis 1 — Coupling sweep: coupling_scale × M structure × manipulations")
    sweep = run_coupling_sweep(
        coupling_scales=[0.0, 0.25, 0.5, 0.75, 1.0],
        m_structures=["random_sparse", "dense", "rank1"],
        seeds=SEEDS,
        base_params=BASE,
    )
    for struct in ("random_sparse", "dense", "rank1"):
        sub = sweep[sweep["m_structure"] == struct]
        print(f"\n--- M structure: {struct} ---")
        for manip in ("none", "magnitude", "orientation"):
            s = sub[sub["manipulation"] == manip].sort_values("coupling_scale")
            print(f"\n  manipulation = {manip}")
            print(
                s[["coupling_scale", "delta_mean", "delta_std", "angle_mean", "angle_std"]]
                .to_string(index=False)
            )
    fig = plot_coupling_sweep(sweep)
    fig.savefig("build/rung4_coupling_sweep.png", dpi=150, bbox_inches="tight")
    print("\n  [saved build/rung4_coupling_sweep.png]")


def axis2_analytic_comparison() -> None:
    _header("Axis 2 — Analytic vs PCA-measured angle (orientation arm)")
    comparison = run_analytic_comparison(
        coupling_scales=[0.0, 0.25, 0.5, 0.75, 1.0],
        m_structures=["random_sparse", "dense", "rank1"],
        seeds=SEEDS,
        base_params=BASE,
    )
    for struct in ("random_sparse", "dense", "rank1"):
        sub = (
            comparison[comparison["m_structure"] == struct]
            .groupby("coupling_scale")
            .agg(
                angle_pred_mean=("angle_pred", "mean"),
                angle_pred_std=("angle_pred", "std"),
                angle_meas_mean=("angle_meas", "mean"),
                angle_meas_std=("angle_meas", "std"),
                angle_anchor_mean=("angle_anchor", "mean"),
            )
            .reset_index()
        )
        print(f"\n--- M structure: {struct} ---")
        print(sub.to_string(index=False))
    fig = plot_analytic_comparison(comparison)
    fig.savefig("build/rung4_analytic_comparison.png", dpi=150, bbox_inches="tight")
    print("\n  [saved build/rung4_analytic_comparison.png]")


def axis3_dim_ratio_sweep() -> None:
    _header("Axis 3 — dim_ratio sweep at coupling_scale=0.75 (random_sparse)")
    result = run_dim_ratio_sweep(
        dim_ratios=[0.5, 1.0, 5.0],
        coupling_scale=0.75,
        seeds=SEEDS,
        base_params=BASE,
    )
    for manip in ("none", "magnitude", "orientation"):
        sub = result[result["manipulation"] == manip].sort_values("dim_ratio")
        print(f"\n  manipulation = {manip}")
        print(
            sub[["dim_ratio", "delta_mean", "delta_std", "angle_mean", "angle_std"]]
            .to_string(index=False)
        )


def axis4_matrix_seed_sweep() -> None:
    _header("Axis 4 — Matrix-seed stability (coupling_scale=0.75, random_sparse)")
    result = run_matrix_seed_sweep(
        matrix_seeds=list(range(5)),
        coupling_scale=0.75,
        seeds=SEEDS,
        base_params=BASE,
    )
    for manip in ("none", "magnitude", "orientation"):
        sub = result[result["manipulation"] == manip].sort_values("matrix_seed")
        print(f"\n  manipulation = {manip}")
        print(
            sub[["matrix_seed", "delta_mean", "delta_std", "angle_mean", "angle_std"]]
            .to_string(index=False)
        )


if __name__ == "__main__":
    os.makedirs("build", exist_ok=True)

    print("Rung-4 cross-omic coupling probe")
    print(
        f"BASE: n_features_anchor={BASE.n_features_anchor}, "
        f"n_samples_per_cell={BASE.n_samples_per_cell}, "
        f"signal_scale={BASE.signal_scale}, "
        f"noise_scale={BASE.noise_scale}, "
        f"dim_ratio={BASE.dim_ratio}, "
        f"nnz_per_nuis={BASE.nnz_per_nuis}, "
        f"n_components={BASE.n_components}"
    )
    print(f"Seeds: {SEEDS}")

    axis1_coupling_sweep()
    axis2_analytic_comparison()
    axis3_dim_ratio_sweep()
    axis4_matrix_seed_sweep()

    print("\nDone.")
    sys.exit(0)
