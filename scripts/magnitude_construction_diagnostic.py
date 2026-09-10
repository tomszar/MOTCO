#!/usr/bin/env python3
"""Magnitude-construction diagnostic (numpy generator, no R, no cluster).

Answers the three questions the Phase 5 exit review needs after the paper-grade
run returned HOLD on the two magnitude mandatory controls:

1. **Where does the off-target response live?** Reads the per-omic scopes already
   persisted in a merged record set and reports, per checkpoint and statistic,
   whether the response exists inside a single omic block or only once the blocks
   are concatenated (``joint_only``).
2. **Is the localization instrument able to classify it?** Runs
   ``localize_off_diagonal`` under both the legacy absolute-threshold rule and
   the null-dispersion rule, side by side.
3. **Is a size-pure magnitude change constructible?** Compares the production
   construction (methylation's delta only) against the diagnostic uniform-delta
   probe (every omic's delta together) on population-standardized geometry, at
   four stages and in the shape-free two-stage regime.

Steps 1 and 2 read committed records and cost nothing. Step 3 generates data but
computes analytic population geometry only — no sampling, no RRPP, no PLS fit.

Example:

    python scripts/magnitude_construction_diagnostic.py \\
        --merged results/phase5-2026-09-10/merged.jsonl \\
        --out-dir results/magnitude-construction-2026-09-10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from motco.simulations.grid import read_replicate_results
from motco.simulations.specificity import (
    compare_uniform_delta_construction,
    decompose_block_response,
    summarize_block_localization,
)
from motco.simulations.study.phase4 import localize_off_diagonal

MODES = ("magnitude", "orientation", "shape")


def _block_tables(records) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail: list[pd.DataFrame] = []
    summary: list[pd.DataFrame] = []
    for mode in MODES:
        try:
            frame = decompose_block_response(records, mode=mode)
        except ValueError as exc:  # a mode absent from this record set
            print(f"  {mode}: skipped ({exc})")
            continue
        detail.append(frame)
        block = summarize_block_localization(frame)
        block.insert(0, "trajectory_mode", mode)
        summary.append(block)
    if not detail:
        raise SystemExit("No trajectory mode in the record set could be decomposed.")
    return pd.concat(detail, ignore_index=True), pd.concat(summary, ignore_index=True)


def _localization_tables(records) -> pd.DataFrame:
    frames = []
    for rule in ("absolute", "null_dispersion"):
        frame = localize_off_diagonal(records, rule=rule)
        if "materiality_rule" not in frame.columns:
            frame = frame.assign(materiality_rule=rule)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _uniform_tables(args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for n_stages in (4, 2):
        for row in compare_uniform_delta_construction(
            effect_sizes=tuple(args.effect_sizes),
            n_samples=args.n_samples,
            n_stages=n_stages,
            p_dmp=args.p_dmp,
            seed=args.seed,
        ):
            record = dict(row.__dict__)
            record["n_stages"] = n_stages
            rows.append(record)
    frame = pd.DataFrame(rows)
    ordered = [
        "n_stages",
        "construction",
        "effect_size",
        "joint_delta",
        "joint_angle",
        "joint_shape",
        "max_block_angle",
        "max_block_shape",
    ]
    return frame[ordered]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--merged",
        type=Path,
        default=Path("results/phase5-2026-09-10/merged.jsonl"),
        help="Merged JSONL whose per-omic geometry is decomposed.",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for the CSV outputs.")
    parser.add_argument("--n-samples", type=int, default=1200, help="Design-point sample size.")
    parser.add_argument("--p-dmp", type=float, default=0.1, help="Design-point differential proportion.")
    parser.add_argument("--seed", type=int, default=2, help="Generator seed for the comparator.")
    parser.add_argument(
        "--effect-sizes",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="Effect grid for the uniform-delta comparator.",
    )
    parser.add_argument(
        "--skip-comparator",
        action="store_true",
        help="Only read committed records (steps 1-2); skip the generating step.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.merged.exists():
        raise SystemExit(f"Merged JSONL not found: {args.merged}")
    print(f"Reading {args.merged} ...")
    records = read_replicate_results(args.merged)
    print(f"  {len(records)} records")

    print("Block decomposition ...")
    detail, summary = _block_tables(records)
    detail.to_csv(out_dir / "block_decomposition.csv", index=False)
    summary.to_csv(out_dir / "block_localization_summary.csv", index=False)
    joint_only = summary[summary["joint_only"]]
    for row in joint_only.itertuples(index=False):
        print(
            f"  joint-only: {row.trajectory_mode}/{row.statistic} at {row.checkpoint} "
            f"(blocks {row.max_block_response:.3g}, joint {row.joint_response:.3g})"
        )
    if joint_only.empty:
        print("  no joint-only response found")

    print("Localization under both materiality rules ...")
    localization = _localization_tables(records)
    localization.to_csv(out_dir / "localization_by_rule.csv", index=False)
    for rule, group in localization.groupby("materiality_rule", sort=True):
        immaterial = group[group["classification"] == "not_material"]
        print(f"  {rule}: {len(immaterial)} of {len(group)} pairs report not_material")

    if not args.skip_comparator:
        print("Uniform-delta comparator (population geometry only) ...")
        uniform = _uniform_tables(args)
        uniform.to_csv(out_dir / "uniform_delta_comparison.csv", index=False)
        top = uniform[(uniform["effect_size"] == max(args.effect_sizes)) & (uniform["n_stages"] == 4)]
        for row in top.itertuples(index=False):
            print(
                f"  {row.construction}: joint delta {row.joint_delta:.3g}, "
                f"joint angle {row.joint_angle:.3g}, joint shape {row.joint_shape:.3g}"
            )

    print(f"\nOutputs under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
