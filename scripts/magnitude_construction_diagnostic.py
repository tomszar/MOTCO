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
   construction (``magnitude_kind='all'``, methylation's delta only) against the
   joint construction (``magnitude_kind='joint'``, every omic's delta together)
   on population-standardized geometry, at four stages and in the shape-free
   two-stage regime.

Steps 1 and 2 read committed records and cost nothing. Step 3 generates data but
computes analytic population geometry only — no sampling, no RRPP, no PLS fit.

``--bracket`` is a fourth, stand-alone mode: it runs step 3's population path
for ``all`` and ``joint`` over a dense effect grid (0 to 1 in steps of 0.01,
plus any extra points given) at the Phase 5 design point and writes
``effect_axis_bracket.csv`` so a magnitude study's effect grid can be chosen
from the *realized* joint ``delta`` rather than the nominal effect. It reads no
records and needs no ``--merged`` file.

Examples:

    python scripts/magnitude_construction_diagnostic.py \\
        --merged results/phase5-2026-09-10/merged.jsonl \\
        --out-dir results/magnitude-construction-2026-09-10

    python scripts/magnitude_construction_diagnostic.py --bracket \\
        --out-dir results/magnitude-axis-bracket-2026-10-08
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
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


BRACKET_CONSTRUCTIONS = ("all", "joint")


def _bracket_grid(step: float, extra: list[float]) -> list[float]:
    dense = np.round(np.arange(0.0, 1.0 + step / 2, step), 6).tolist()
    return sorted(set(dense) | {round(float(e), 6) for e in extra})


def _bracket_table(args: argparse.Namespace) -> pd.DataFrame:
    grid = _bracket_grid(args.bracket_step, args.effect_sizes)
    rows = [
        dict(row.__dict__)
        for row in compare_uniform_delta_construction(
            effect_sizes=tuple(grid),
            n_samples=args.n_samples,
            n_stages=args.n_stages,
            p_dmp=args.p_dmp,
            seed=args.seed,
            magnitude_kinds=BRACKET_CONSTRUCTIONS,
        )
    ]
    frame = pd.DataFrame(rows)
    all_delta = frame[frame["construction"] == "all"].set_index("effect_size")["joint_delta"]
    denominator = frame["effect_size"].map(all_delta).to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        frame["ratio_to_all"] = np.where(
            denominator > 0, frame["joint_delta"].to_numpy(dtype=float) / denominator, np.nan
        )
    ordered = [
        "construction",
        "effect_size",
        "joint_delta",
        "joint_angle",
        "joint_shape",
        "ratio_to_all",
        "max_block_angle",
        "max_block_shape",
    ]
    return frame[ordered].sort_values(["construction", "effect_size"]).reset_index(drop=True)


def _run_bracket(args: argparse.Namespace) -> int:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Effect-axis bracket (population geometry only; n_samples={args.n_samples}, "
        f"n_stages={args.n_stages}, p_dmp={args.p_dmp}, seed={args.seed}) ..."
    )
    bracket = _bracket_table(args)
    path = out_dir / "effect_axis_bracket.csv"
    bracket.to_csv(path, index=False)
    anchor = bracket[bracket["effect_size"] == 0.0].iloc[0]
    print(f"  anchor floor: angle {anchor.joint_angle:.3g}, shape {anchor.joint_shape:.3g}")
    joint = bracket[bracket["construction"] == "joint"]
    print(
        f"  joint: angle max {joint['joint_angle'].abs().max():.3g}, "
        f"shape max {joint['joint_shape'].abs().max():.3g} over {len(joint)} effects"
    )
    for effect in sorted(set(args.effect_sizes)):
        at = bracket[bracket["effect_size"] == round(float(effect), 6)].set_index("construction")
        print(
            f"  e={effect:g}: all delta {at.loc['all', 'joint_delta']:.3g}, "
            f"joint delta {at.loc['joint', 'joint_delta']:.3g} "
            f"(ratio {at.loc['joint', 'ratio_to_all']:.3g})"
        )
    print(f"\nWrote {path}")
    return 0


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
    parser.add_argument("--n-stages", type=int, default=4, help="Design-point stage count (bracket only).")
    parser.add_argument(
        "--effect-sizes",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="Effect grid for the uniform-delta comparator; under --bracket, extra points "
        "added to the dense grid and echoed in the summary.",
    )
    parser.add_argument(
        "--bracket",
        action="store_true",
        help="Write the analytic effect-axis bracket (effect_axis_bracket.csv) and exit; "
        "reads no records.",
    )
    parser.add_argument(
        "--bracket-step",
        type=float,
        default=0.01,
        help="Dense-grid step for --bracket (default: 0.01 from 0 to 1).",
    )
    parser.add_argument(
        "--skip-comparator",
        action="store_true",
        help="Only read committed records (steps 1-2); skip the generating step.",
    )
    args = parser.parse_args()

    if args.bracket:
        return _run_bracket(args)

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
        print("Magnitude-construction comparator (population geometry only) ...")
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
