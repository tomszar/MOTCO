#!/usr/bin/env python3
"""Driver script for the Rung-2 projector recovery probe.

Runs the per-projector comparison, the dimensionality and effect-size sweeps, and
the supervised-leakage probe; prints the summary tables and saves the
per-projector comparison figure. Also reports the standardize arm under
anisotropic noise (where raw PCA degrades but standardization restores recovery).

Usage
-----
    .venv/bin/python scripts/projector_recovery_probe.py
    .venv/bin/python scripts/projector_recovery_probe.py \\
        --signal_scale 5 --noise_scale 1 --n_components 2 --n_seeds 10
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from motco.simulations.projector_recovery import (
    ProjectorRecoveryParams,
    plot_projector_comparison,
    run_dimensionality_sweep,
    run_effect_size_sweep,
    run_leakage_probe,
    run_projector_comparison,
)

BUILD = Path(__file__).resolve().parent.parent / "build"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Rung-2 projector recovery probe",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n_features", type=int, default=50)
    p.add_argument("--n_components", type=int, default=2)
    p.add_argument("--signal_scale", type=float, default=5.0)
    p.add_argument("--noise_scale", type=float, default=1.0)
    p.add_argument("--n_samples_per_cell", type=int, default=40)
    p.add_argument("--scale_c", type=float, default=2.0)
    p.add_argument("--angle_theta", type=float, default=45.0)
    p.add_argument("--anisotropy", type=float, default=1.5, help="Aniso sweep value")
    p.add_argument("--n_seeds", type=int, default=10, help="Seeds to average over")
    p.add_argument(
        "--component_grid",
        type=int,
        nargs="+",
        default=[2, 3, 5, 10],
        help="Latent component counts to sweep",
    )
    p.add_argument(
        "--signal_scales",
        type=float,
        nargs="+",
        default=[2.0, 5.0, 8.0, 12.0],
        help="Effect sizes to sweep",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for the figure (default: build/rung2_projector_comparison.png)",
    )
    return p


def main() -> None:
    args = _parser().parse_args()
    base = ProjectorRecoveryParams(
        n_features=args.n_features,
        n_components=args.n_components,
        signal_scale=args.signal_scale,
        noise_scale=args.noise_scale,
        n_samples_per_cell=args.n_samples_per_cell,
        scale_c=args.scale_c,
        angle_theta=args.angle_theta,
    )
    seeds = list(range(args.n_seeds))
    fmt = "{:.3f}".format
    cols = ["projector", "manipulation", "delta_mean", "delta_std", "angle_mean", "angle_std"]

    print(
        f"Projector recovery probe  |  n_features={args.n_features}  "
        f"k={args.n_components}  signal={args.signal_scale}  noise={args.noise_scale}  "
        f"n_per_cell={args.n_samples_per_cell}  seeds={seeds}"
    )
    print()

    print("=== Per-projector comparison (isotropic noise) ===")
    comp = run_projector_comparison(seeds=seeds, base_params=base)
    print(comp[cols].to_string(index=False, float_format=fmt))
    print()

    print(f"=== Per-projector comparison (anisotropic noise = {args.anisotropy}) ===")
    comp_aniso = run_projector_comparison(
        seeds=seeds, base_params=replace(base, anisotropy=args.anisotropy)
    )
    print(comp_aniso[cols].to_string(index=False, float_format=fmt))
    print()

    print("=== Dimensionality sweep ===")
    dim = run_dimensionality_sweep(args.component_grid, seeds=seeds, base_params=base)
    print(
        dim[["n_components", *cols]].to_string(index=False, float_format=fmt)
    )
    print()

    print("=== Effect-size sweep ===")
    eff = run_effect_size_sweep(args.signal_scales, seeds=seeds, base_params=base)
    print(eff[["signal_scale", *cols]].to_string(index=False, float_format=fmt))
    print()

    print("=== Supervised-leakage probe (none trajectory: PLS-DA vs PCA) ===")
    leak = run_leakage_probe(args.component_grid, seeds=seeds, base_params=base)
    print(leak.to_string(index=False, float_format=fmt))
    print()

    print("Generating per-projector comparison figure ...")
    fig = plot_projector_comparison(comp)
    out = args.output
    if out is None:
        BUILD.mkdir(parents=True, exist_ok=True)
        out = BUILD / "rung2_projector_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure saved → {out}")


if __name__ == "__main__":
    main()
