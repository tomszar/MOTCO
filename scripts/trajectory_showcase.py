#!/usr/bin/env python3
"""Render the trajectory-mode showcase figure (numpy generator, no R).

Generates one semi-synthetic dataset per ``trajectory_mode`` from a shared
baseline (cached reference data), projects each through a 2-component PLS-DA
(stage as the response, no cross-validation), and saves a multi-panel
comparison figure.

Example:

    python scripts/trajectory_showcase.py \
        --seed 0 --n-sample 1000 --effect-size 1.0 \
        --out /tmp/trajectory_showcase.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from motco.simulations.showcase import TRAJECTORY_SHOWCASE_MODES, run_trajectory_showcase


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trajectory_showcase", description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for generation and group assignment.")
    parser.add_argument("--n-sample", type=int, default=1000, help="Total number of samples (default: 1000).")
    parser.add_argument("--n-stages", type=int, default=3, help="Number of trajectory stages (default: 3).")
    parser.add_argument(
        "--effect-size",
        type=float,
        default=1.0,
        help="Injected group effect size for the non-null modes (default: 1.0).",
    )
    parser.add_argument(
        "--p-dmp",
        type=float,
        default=0.2,
        help="Per-stage probability a methylation feature is differential (default: 0.2).",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=list(TRAJECTORY_SHOWCASE_MODES),
        help=f"Trajectory modes to render (default: {' '.join(TRAJECTORY_SHOWCASE_MODES)}).",
    )
    parser.add_argument(
        "--no-samples",
        action="store_true",
        help="Do not overlay individual sample points behind the trajectories.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output figure path (e.g. .png or .pdf).")
    parser.add_argument("--dpi", type=int, default=150, help="Figure resolution in dots per inch (default: 150).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fig, datasets = run_trajectory_showcase(
        seed=args.seed,
        n_sample=args.n_sample,
        n_stages=args.n_stages,
        effect_size=args.effect_size,
        modes=tuple(args.modes),
        p_dmp=args.p_dmp,
        show_samples=not args.no_samples,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"Wrote {args.out} ({len(datasets)} scenarios: {', '.join(datasets)}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
