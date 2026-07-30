#!/usr/bin/env python3
"""Run the controlled inverse-PLS trajectory interpretability study."""

from __future__ import annotations

import argparse
from pathlib import Path

from motco.simulations.pls_inverse import (
    PLSInverseStudyParams,
    render_markdown,
    results_frames,
    run_inverse_study,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the study command-line parser."""

    parser = argparse.ArgumentParser(
        description="Map controlled PLS trajectory interventions back to implied feature changes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-features", type=int, default=50)
    parser.add_argument("--n-samples-per-stage", type=int, default=30)
    parser.add_argument("--noise-scale", type=float, default=0.5)
    parser.add_argument("--signal-scale", type=float, default=4.0)
    parser.add_argument("--magnitude-scale", type=float, default=2.0)
    parser.add_argument("--orientation-degrees", type=float, default=45.0)
    parser.add_argument("--shape-displacement-fraction", type=float, default=0.5)
    parser.add_argument("--out-dir", type=Path, default=Path("build/pls_inverse_interpretability"))
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run all five cells and write CSV and Markdown outputs."""

    args = build_parser().parse_args(argv)
    params = PLSInverseStudyParams(
        seed=args.seed,
        n_features=args.n_features,
        n_samples_per_stage=args.n_samples_per_stage,
        noise_scale=args.noise_scale,
        signal_scale=args.signal_scale,
        magnitude_scale=args.magnitude_scale,
        orientation_degrees=args.orientation_degrees,
        shape_displacement_fraction=args.shape_displacement_fraction,
    )
    results = run_inverse_study(params)
    cells, features = results_frames(results)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cells.to_csv(args.out_dir / "cells.csv", index=False)
    features.to_csv(args.out_dir / "features.csv", index=False)
    report = render_markdown(results)
    (args.out_dir / "report.md").write_text(report)
    print(report)
    print(f"Wrote results to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
