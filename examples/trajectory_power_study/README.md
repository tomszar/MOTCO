# Trajectory power study

A reproducible, sharded study that characterizes the Type I error and power of the
MOTCO trajectory test (delta, angle, shape) under semi-synthetic InterSIM datasets.

## Workflow

```
study config (YAML/JSON)
    │
    ▼  enumerate_study → SimulationGrid (Type I + power cells)
    │
    ▼  shard runner: scripts/run_study_shard.py
    │     (one cluster array task per shard, writes shard_<i>.jsonl)
    │
    ▼  merge: python scripts/motco_study.py merge --out-dir <dir>
    │     (combines shards into merged.jsonl, dedup by (cell, replicate))
    │
    ▼  report: python scripts/motco_study.py report --config <cfg> --out-dir <dir>
          (per-statistic + combined-rule summaries → specificity matrix,
           power curves, Type I table; CSV + PNG; acceptance-target report)
```

## Local smoke run

```bash
# Generate a few shards locally (no cluster):
python scripts/run_study_shard.py \
    --config examples/trajectory_power_study/smoke.json \
    --out-dir /tmp/motco-smoke \
    --shard-index 0 --n-shards 4 --error-policy record
python scripts/run_study_shard.py \
    --config examples/trajectory_power_study/smoke.json \
    --out-dir /tmp/motco-smoke \
    --shard-index 1 --n-shards 4 --error-policy record
# (repeat for shards 2 and 3, or run with --n-shards 1 to do them all locally)

python scripts/motco_study.py merge --out-dir /tmp/motco-smoke
python scripts/motco_study.py report \
    --config examples/trajectory_power_study/smoke.json \
    --out-dir /tmp/motco-smoke
```

Outputs land under `/tmp/motco-smoke/report/`:

- `specificity_matrix.csv` / `.png` — mode × statistic rejection rates
- `power_curves.csv` / `.png` — per-statistic rejection rate vs effect size
- `type_i_table.csv` / `type_i.png` — null-cell per-statistic + combined-rule rates
- `acceptance_report.csv` / `.json` — pre-specified targets evaluated against
  observed Monte Carlo uncertainty (non-gating)

## SLURM cluster run

```bash
# Submit an array of size N_SHARDS:
sbatch \
    --array=0-63 \
    --export=ALL,STUDY_CONFIG=$(pwd)/examples/trajectory_power_study/smoke.json,STUDY_OUT=$(pwd)/results,N_SHARDS=64 \
    scripts/motco_study_array.sbatch

# After completion:
python scripts/motco_study.py merge  --out-dir results
python scripts/motco_study.py report --config examples/trajectory_power_study/smoke.json --out-dir results
```

Failed array tasks can be resubmitted with `--array=7,12,40` — the shard-resume
guard (parameter signature) skips already-completed replicates.

## Config quick reference

| Field             | Purpose                                                    |
|-------------------|------------------------------------------------------------|
| `intersim`        | Baseline InterSIM params (R generator)                     |
| `generator`       | Baseline semi-synthetic perturbation params                |
| `evaluation`      | Integration method, RRPP permutations, n_jobs              |
| `trajectory_modes`| Power-grid modes (e.g. `magnitude`, `orientation`, …)      |
| `effect_sizes`    | Power-grid effect sizes                                    |
| `axes`            | OFAT axes, namespaced `intersim.` / `generator.` / `evaluation.` |
| `n_replicates`    | Replicates per cell                                        |
| `base_seed`       | Deterministic seed root                                    |
| `alpha`           | Significance level for rejection rates                     |
| `acceptance`      | Pre-specified Type I, power, and specificity targets       |

`none` is always present as the Type I baseline (enforced by enumeration);
`translation` is added explicitly as a second negative control.
