#!/usr/bin/env python3
"""Angle-pivotality diagnostic over merged study records.

Answers one question from records alone, with no additional permutations: does
each replicate's RRPP permutation null move with its own observed statistic? A
non-pivotal statistic does not break validity — RRPP p-values stay exact — it
costs power, because a replicate whose latent geometry inflates the observed
statistic inflates its own critical value with it.

Reads a merged JSONL record set (records must carry the per-replicate null
summary; see ``motco.simulations.grid.SimulationReplicateResult``) and writes
four tables:

1. ``pivotality_association.csv`` — per cell and statistic, the correlation and
   slope of the null's mean, spread, and q95 against the observed statistic,
   with a Fisher-z interval on each correlation.
2. ``pivotality_rejection_split.csv`` — the observed statistic and its critical
   value split by rejection outcome, flagging the Phase 4 inversion.
3. ``pivotality_standardized.csv`` — the as-specified rejection rate beside the
   cross-replicate standardized rate, for every cell including the controls.
4. ``pivotality_spectrum.csv`` — per cell and statistic, the association between
   the recorded latent configuration eigengap and the width of that replicate's
   own null. Reported as ``status='unavailable'`` for records predating the
   spectrum field.

The standardized counterfactual is a **diagnostic, not a deployable test**: it
borrows a reference ``z`` distribution from the null-control cells, which does
not exist in real data. Within-replicate studentization is a no-op — it rescales
both sides of the comparison and leaves the p-value unchanged.

Example:

    python scripts/angle_null_pivotality.py \
        --merged results/angle-pivotality-2026-09-01/merged.jsonl \
        --out-dir results/angle-pivotality-2026-09-01/report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from motco.simulations.grid import read_replicate_results
from motco.simulations.pivotality import (
    STATISTICS,
    association_table,
    group_by_cell,
    spectrum_association_table,
    write_pivotality_tables,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="angle_null_pivotality",
        description="Measure whether each replicate's RRPP null tracks its own observed statistic.",
    )
    parser.add_argument("--merged", type=Path, required=True, help="Merged JSONL record set.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for the CSV tables.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Rejection level (default: 0.05).")
    parser.add_argument(
        "--statistics",
        nargs="+",
        default=list(STATISTICS),
        help=f"Statistics to analyze (default: {' '.join(STATISTICS)}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.merged.exists():
        print(f"error: no such record set: {args.merged}", file=sys.stderr)
        return 2
    records = read_replicate_results(args.merged)
    if not records:
        print(f"error: {args.merged} holds no records", file=sys.stderr)
        return 2

    completed = [record for record in records if record.status == "completed"]
    with_summary = [record for record in completed if record.null_summary]
    print(f"records: {len(records)} total, {len(completed)} completed, {len(with_summary)} with a null summary")
    if not with_summary:
        print(
            "error: no completed record carries a null summary — these records predate the field, "
            "so the pivotality question cannot be answered without a re-run.",
            file=sys.stderr,
        )
        return 2

    for cell, group in group_by_cell(completed).items():
        print(f"  {cell.label}: {len(group)} completed")

    paths = write_pivotality_tables(
        completed, args.out_dir, alpha=args.alpha, statistics=tuple(args.statistics)
    )
    print("\nwrote:")
    for name, path in paths.items():
        print(f"  {name}: {path}")

    print("\nnull-tracking correlation (observed vs its own q95):")
    for row in association_table(completed, statistics=tuple(args.statistics), null_targets=("q95",)):
        correlation = "n/a" if row.correlation is None else f"{row.correlation:+.3f}"
        interval = (
            ""
            if row.correlation_ci_low is None or row.correlation_ci_high is None
            else f" [{row.correlation_ci_low:+.3f}, {row.correlation_ci_high:+.3f}]"
        )
        slope = "n/a" if row.slope is None else f"{row.slope:+.3f}"
        print(f"  {row.cell.label:<38} {row.statistic:<6} r={correlation}{interval}  slope={slope}")

    print("\neigengap vs its own null width (q95), per cell:")
    for row in spectrum_association_table(
        completed, statistics=tuple(args.statistics), null_targets=("q95",)
    ):
        if row.status != "ok":
            print(f"  {row.cell.label:<38} {row.statistic:<6} unavailable (no spectrum recorded)")
            continue
        spearman = "n/a" if row.spearman is None else f"{row.spearman:+.3f}"
        interval = (
            ""
            if row.spearman_ci_low is None or row.spearman_ci_high is None
            else f" [{row.spearman_ci_low:+.3f}, {row.spearman_ci_high:+.3f}]"
        )
        pearson = "n/a" if row.log_log_pearson is None else f"{row.log_log_pearson:+.3f}"
        print(
            f"  {row.cell.label:<38} {row.statistic:<6} spearman={spearman}{interval}"
            f"  log-log r={pearson}  n={row.n_replicates}"
        )

    print(
        "\nThe standardized counterfactual is a diagnostic, not a deployable test: it borrows a\n"
        "reference z distribution from the null-control cells. Within-replicate studentization\n"
        "rescales both sides of the comparison and leaves the p-value unchanged."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
