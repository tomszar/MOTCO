#!/usr/bin/env python3
"""Driver script for the Rung-0 Gaussian existence proof.

Runs the existence-proof driver over a configurable set of seeds, prints the
delta / angle summary table, and saves a 3-panel latent-space trajectory figure.

Usage
-----
    uv run python scripts/linear_recovery_probe.py
    uv run python scripts/linear_recovery_probe.py \\
        --seed 42 --n_features 20 --n_components 15 --signal_scale 8 --n_seeds 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

from motco.simulations.linear_recovery import (
    LinearRecoveryParams,
    plot_latent_trajectories,
    run_existence_proof,
)

BUILD = Path(__file__).resolve().parent.parent / "build"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Rung-0 Gaussian existence proof driver",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--seed", type=int, default=0, help="Base RNG seed")
    p.add_argument("--n_features", type=int, default=50)
    p.add_argument("--n_components", type=int, default=10)
    p.add_argument("--signal_scale", type=float, default=5.0)
    p.add_argument("--noise_scale", type=float, default=1.0)
    p.add_argument("--n_samples_per_cell", type=int, default=40)
    p.add_argument("--scale_c", type=float, default=2.0)
    p.add_argument("--angle_theta", type=float, default=45.0)
    p.add_argument("--n_seeds", type=int, default=10, help="Seeds to average over")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for the trajectory figure (default: build/rung0_latent_trajectories.png)",
    )
    return p


def main() -> None:
    args = _parser().parse_args()

    base_params = LinearRecoveryParams(
        seed=args.seed,
        n_features=args.n_features,
        n_components=args.n_components,
        signal_scale=args.signal_scale,
        noise_scale=args.noise_scale,
        n_samples_per_cell=args.n_samples_per_cell,
        scale_c=args.scale_c,
        angle_theta=args.angle_theta,
    )

    seeds = list(range(args.n_seeds))

    print(
        f"Existence proof  |  n_features={args.n_features}  "
        f"n_components={args.n_components}  "
        f"signal={args.signal_scale}  noise={args.noise_scale}  "
        f"n_per_cell={args.n_samples_per_cell}  seeds={seeds}"
    )
    print()

    table = run_existence_proof(seeds=seeds, base_params=base_params)
    print(table.to_string(index=False, float_format="{:.3f}".format))
    print()

    print("Generating trajectory figure ...")
    fig = plot_latent_trajectories(
        seed=args.seed, base_params=base_params, show_samples=True
    )

    out = args.output
    if out is None:
        BUILD.mkdir(parents=True, exist_ok=True)
        out = BUILD / "rung0_latent_trajectories.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure saved → {out}")


if __name__ == "__main__":
    main()
