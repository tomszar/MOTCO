#!/usr/bin/env python3
"""Driver script for the Rung-1 methylation rev.logit recovery probe.

Runs the operating-point sweep and the step-scale sweep, prints both summary
tables, and saves the operating-point distortion figure.

Usage
-----
    .venv/bin/python scripts/methylation_recovery_probe.py
    .venv/bin/python scripts/methylation_recovery_probe.py \\
        --signal_scale 2 --noise_scale 0.3 --n_components 2 --n_seeds 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

from motco.simulations.methylation_recovery import (
    MethylationRecoveryParams,
    plot_operating_point_sweep,
    run_integration_contrast,
    run_operating_point_sweep,
    run_step_scale_sweep,
)

BUILD = Path(__file__).resolve().parent.parent / "build"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Rung-1 methylation rev.logit recovery probe",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n_features", type=int, default=50)
    p.add_argument("--n_components", type=int, default=2)
    p.add_argument("--signal_scale", type=float, default=2.0)
    p.add_argument("--noise_scale", type=float, default=0.3)
    p.add_argument("--n_samples_per_cell", type=int, default=40)
    p.add_argument("--scale_c", type=float, default=2.0)
    p.add_argument("--angle_theta", type=float, default=45.0)
    p.add_argument("--n_seeds", type=int, default=10, help="Seeds to average over")
    p.add_argument(
        "--m_baselines",
        type=float,
        nargs="+",
        default=[0.0, 1.0, 2.0, 3.0, 4.0],
        help="Operating points to sweep",
    )
    p.add_argument(
        "--signal_scales",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 4.0, 6.0, 8.0],
        help="Step scales for the center step-span sweep",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for the figure (default: build/rung1_operating_point.png)",
    )
    return p


def main() -> None:
    args = _parser().parse_args()
    # The operating-point and step-scale sweeps characterize the β-frame failure
    # mode, so pin them to β-integration; the contrast section runs both arms.
    base = MethylationRecoveryParams(
        n_features=args.n_features,
        n_components=args.n_components,
        signal_scale=args.signal_scale,
        noise_scale=args.noise_scale,
        n_samples_per_cell=args.n_samples_per_cell,
        scale_c=args.scale_c,
        angle_theta=args.angle_theta,
        m_baseline=0.0,
        integration_space="beta",
    )
    seeds = list(range(args.n_seeds))
    fmt = "{:.3f}".format

    print(
        f"Methylation rev.logit probe  |  n_features={args.n_features}  "
        f"k={args.n_components}  signal={args.signal_scale}  "
        f"noise={args.noise_scale}  n_per_cell={args.n_samples_per_cell}  seeds={seeds}"
    )
    print()

    print("=== Operating-point sweep (step scale fixed) ===")
    op = run_operating_point_sweep(args.m_baselines, seeds=seeds, base_params=base)
    print(op.to_string(index=False, float_format=fmt))
    print()

    print("=== Step-scale sweep (operating point fixed at center m=0) ===")
    ss = run_step_scale_sweep(args.signal_scales, seeds=seeds, base_params=base)
    print(
        ss[
            [
                "signal_scale",
                "manipulation",
                "delta_mean",
                "delta_std",
                "angle_mean",
                "angle_std",
            ]
        ].to_string(index=False, float_format=fmt)
    )
    print()

    print("=== Integration-space contrast: β vs M-value (step-scale sweep) ===")
    contrast = run_integration_contrast(
        args.signal_scales, seeds=seeds, base_params=base
    )
    print(
        contrast[
            [
                "integration_space",
                "signal_scale",
                "manipulation",
                "delta_mean",
                "angle_mean",
                "angle_std",
            ]
        ].to_string(index=False, float_format=fmt)
    )
    print()

    print("Generating operating-point figure ...")
    fig = plot_operating_point_sweep(op)
    out = args.output
    if out is None:
        BUILD.mkdir(parents=True, exist_ok=True)
        out = BUILD / "rung1_operating_point.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure saved → {out}")


if __name__ == "__main__":
    main()
