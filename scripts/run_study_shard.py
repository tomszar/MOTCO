#!/usr/bin/env python3
"""Run one shard of a MOTCO trajectory power study.

Typical use on a cluster (driven by motco_study_array.sbatch):

    python scripts/run_study_shard.py \
        --config /abs/path/study.yaml \
        --out-dir /abs/path/out \
        --shard-index "$SLURM_ARRAY_TASK_ID" \
        --n-shards "$N_SHARDS" \
        --n-jobs "$SLURM_CPUS_PER_TASK"

Each invocation writes/extends `shard_<shard_index>.jsonl` under ``--out-dir`` and
is fully resumable (signature-guarded). Re-running the same shard skips already
completed replicates.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from motco.simulations.study.config import load_study_config
from motco.simulations.study.enumerate import enumerate_study
from motco.simulations.study.sharding import run_shard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_study_shard", description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to the study config (YAML or JSON).")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for shard JSONL files.")
    parser.add_argument("--shard-index", type=int, required=True, help="0-based shard index for this task.")
    parser.add_argument("--n-shards", type=int, required=True, help="Total number of shards across the array.")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Within-replicate RRPP parallelism (CPUs per task). Overrides the config when set.",
    )
    parser.add_argument(
        "--error-policy",
        choices=("raise", "record"),
        default="raise",
        help="Per-replicate failure handling: 'record' continues, 'raise' aborts the shard.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard the existing shard JSONL before running (use with care).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_study_config(args.config)
    if args.n_jobs is not None:
        evaluation = replace(config.evaluation, n_jobs=args.n_jobs)
        config = replace(config, evaluation=evaluation)
    grid = enumerate_study(config)
    records = run_shard(
        grid,
        shard_index=args.shard_index,
        n_shards=args.n_shards,
        out_dir=args.out_dir,
        error_policy=args.error_policy,
        overwrite=args.overwrite,
    )
    completed = sum(1 for r in records if r.status == "completed")
    failed = sum(1 for r in records if r.status == "failed")
    print(
        f"shard {args.shard_index}/{args.n_shards}: {completed} new completed, "
        f"{failed} failed (existing replicates skipped)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
