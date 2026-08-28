#!/bin/bash
#SBATCH --job-name=motco-study
#SBATCH --output=logs/motco-study-%A_%a.out
#SBATCH --error=logs/motco-study-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-N_SHARDS_MINUS_ONE

# MOTCO trajectory power study — SLURM array template.
#
# Usage:
#   sbatch \
#     --export=ALL,STUDY_CONFIG=/abs/path/study.yaml,STUDY_OUT=/abs/path/out,N_SHARDS=64 \
#     --array=0-63 \
#     scripts/motco_study_array.sbatch
#
# Required environment variables (passed via --export):
#   STUDY_CONFIG  Absolute path to the study config (YAML or JSON).
#   STUDY_OUT     Output directory; per-shard `shard_<i>.jsonl` files are written here.
#   N_SHARDS      Total shards = #SBATCH --array width (also passed to the runner).
#
# Optional environment variable:
#   STUDY_N_JOBS  Within-replicate RRPP parallelism. Leave UNSET to use the
#                 value in the config, which is what a reproducible study run
#                 wants.
#
# Why n_jobs is not simply $SLURM_CPUS_PER_TASK:
#   RRPP seeds one RNG stream per worker, so the realized permutation draws —
#   and therefore the p-values — depend on the worker count. n_jobs is part of
#   `evaluation_params` and so enters the cell parameter signature: changing it
#   between submissions makes completed replicates unresumable and makes the run
#   irreproducible from the committed config alone. Parallelize across array
#   tasks (cells), and only set STUDY_N_JOBS when you deliberately accept a
#   different signature for the whole study.
#
# Resubmitting failed shards:
#   The shard JSONL files are signature-guarded and resumable. If specific
#   array task ids failed, resubmit *only* those ids:
#       sbatch --array=7,12,40 --export=ALL,STUDY_CONFIG=...,STUDY_OUT=...,N_SHARDS=64 \
#              scripts/motco_study_array.sbatch
#   Completed replicates are skipped automatically; only missing or failed
#   replicates within the shard are re-executed.

set -euo pipefail

: "${STUDY_CONFIG:?STUDY_CONFIG environment variable is required.}"
: "${STUDY_OUT:?STUDY_OUT environment variable is required.}"
: "${N_SHARDS:?N_SHARDS environment variable is required.}"

mkdir -p "$STUDY_OUT" logs

# Activate the project virtualenv (adjust the path for your cluster layout).
if [ -f .venv/bin/activate ]; then
    # shellcheck source=/dev/null
    source .venv/bin/activate
fi

n_jobs_arg=()
if [ -n "${STUDY_N_JOBS:-}" ]; then
    n_jobs_arg=(--n-jobs "$STUDY_N_JOBS")
fi

python scripts/run_study_shard.py \
    --config "$STUDY_CONFIG" \
    --out-dir "$STUDY_OUT" \
    --shard-index "$SLURM_ARRAY_TASK_ID" \
    --n-shards "$N_SHARDS" \
    "${n_jobs_arg[@]}" \
    --error-policy record
