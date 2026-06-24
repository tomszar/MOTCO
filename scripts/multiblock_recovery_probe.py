"""Rung-3 multi-block concatenation probe driver.

Runs all four analysis axes defined in
``openspec/changes/rung3-multiblock-concatenation/design.md`` and prints
the headline tables to stdout.  Results inform the findings writeup.

Usage::

    .venv/bin/python scripts/multiblock_recovery_probe.py
"""

from __future__ import annotations

import sys

import pandas as pd

from motco.simulations.multiblock_recovery import (
    MultiblockRecoveryParams,
    plot_block_weight_curve,
    plot_dim_ratio_sweep,
    run_block_comparison,
    run_block_weight_curve,
    run_dim_ratio_sweep,
    run_effect_size_sweep,
    run_rho_sweep,
)

pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 120)

SEEDS = list(range(10))
BASE = MultiblockRecoveryParams(
    seed=0,
    n_features_anchor=50,
    n_samples_per_cell=40,
    noise_scale=1.0,
    signal_scale=5.0,
    n_nuisance_blocks=1,
    dim_ratio=1.0,
    rho_nuisance=0.0,
    n_components=10,
)


def _header(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}\n")


def axis1_block_comparison() -> None:
    _header("Axis 1 — Headline block comparison (n_nuisance_blocks × dim_ratio)")
    result = run_block_comparison(seeds=SEEDS, base_params=BASE)
    for n_blocks in sorted(result["n_nuisance_blocks"].unique()):
        sub = result[result["n_nuisance_blocks"] == n_blocks].copy()
        sub = sub.sort_values(["dim_ratio", "manipulation"])
        print(f"\n--- n_nuisance_blocks = {n_blocks} ---")
        tbl = sub.pivot_table(
            index="dim_ratio",
            columns="manipulation",
            values=["delta_mean", "angle_mean"],
        )
        print(tbl.to_string())


def axis2_dim_ratio_sweep() -> None:
    _header("Axis 2 — Primary dim_ratio sweep (n_nuisance_blocks = 1)")
    sweep = run_dim_ratio_sweep(
        dim_ratios=[0.5, 1.0, 2.0, 5.0, 10.0],
        seeds=SEEDS,
        base_params=BASE,
    )
    for manip in ("none", "magnitude", "orientation"):
        sub = sweep[sweep["manipulation"] == manip].sort_values("dim_ratio")
        print(f"\n  manipulation = {manip}")
        print(
            sub[["dim_ratio", "delta_mean", "delta_std", "angle_mean", "angle_std"]]
            .to_string(index=False)
        )
    fig = plot_dim_ratio_sweep(sweep)
    fig.savefig("build/rung3_dim_ratio_sweep.png", dpi=150, bbox_inches="tight")
    print("\n  [saved build/rung3_dim_ratio_sweep.png]")


def axis3_rho_sweep() -> None:
    _header("Axis 3 — Nuisance-block correlation sweep (dim_ratio = 5.0)")
    result = run_rho_sweep(
        rho_values=[0.0, 0.3, 0.7],
        dim_ratio=5.0,
        seeds=SEEDS,
        base_params=BASE,
    )
    for manip in ("none", "magnitude", "orientation"):
        sub = result[result["manipulation"] == manip].sort_values("rho_nuisance")
        print(f"\n  manipulation = {manip}")
        print(
            sub[["rho_nuisance", "delta_mean", "delta_std", "angle_mean", "angle_std"]]
            .to_string(index=False)
        )


def axis4_effect_size_sweep() -> None:
    _header("Axis 4 — Effect-size sweep across dim_ratio (n_nuisance_blocks = 1)")
    result = run_effect_size_sweep(
        signal_scales=[2.0, 5.0, 8.0, 12.0],
        dim_ratios=[0.0, 1.0, 5.0, 10.0],
        seeds=SEEDS,
        base_params=BASE,
    )
    for manip in ("magnitude", "orientation"):
        sub = result[result["manipulation"] == manip]
        print(f"\n  manipulation = {manip}")
        tbl = sub.pivot_table(
            index="signal_scale",
            columns="dim_ratio",
            values=["delta_mean", "angle_mean"],
        )
        print(tbl.to_string())


def axis5_block_weight_curve() -> None:
    _header("Axis 5 — Anchor block-weight fraction vs dim_ratio")
    weight_df = run_block_weight_curve(
        dim_ratios=[0.5, 1.0, 2.0, 5.0, 10.0],
        n_nuisance_blocks=1,
        seeds=list(range(10)),
        base_params=BASE,
    )
    summary = (
        weight_df.groupby("dim_ratio")
        .agg(
            w_mean=("w_anchor", "mean"),
            w_std=("w_anchor", "std"),
            p_naive=("p_anchor_naive", "mean"),
            n_features_total=("n_features_total", "first"),
        )
        .reset_index()
    )
    print(summary.to_string(index=False))
    fig = plot_block_weight_curve(weight_df)
    fig.savefig("build/rung3_block_weight_curve.png", dpi=150, bbox_inches="tight")
    print("\n  [saved build/rung3_block_weight_curve.png]")


if __name__ == "__main__":
    import os

    os.makedirs("build", exist_ok=True)

    print("Rung-3 multi-block concatenation probe")
    print(f"BASE: n_features_anchor={BASE.n_features_anchor}, "
          f"n_samples_per_cell={BASE.n_samples_per_cell}, "
          f"signal_scale={BASE.signal_scale}, "
          f"noise_scale={BASE.noise_scale}, "
          f"n_components={BASE.n_components}")
    print(f"Seeds: {SEEDS}")

    axis1_block_comparison()
    axis2_dim_ratio_sweep()
    axis3_rho_sweep()
    axis4_effect_size_sweep()
    axis5_block_weight_curve()

    print("\nDone.")
    sys.exit(0)
