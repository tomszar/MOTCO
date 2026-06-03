# MOTCO trajectory power study

This package (`motco.simulations.study`) is the engine behind the
reproducible Type I error / power study for the MOTCO trajectory test
(`delta`, `angle`, `shape`). It enumerates a grid of semi-synthetic
datasets from the numpy generator (cached InterSIM reference data, no R
at runtime), runs the full MOTCO pipeline on each replicate, and
produces summary tables, figures, and an acceptance-target report.

This README is the operational handbook: enough for anyone
to run and replicate the study end-to-end, locally or on
a SLURM cluster.

---

## 1. What the study computes

For each cell in the grid × replicate, the runner:

1. Generates aligned methylation, expression, and proteomics matrices
   via the numpy generator and cached InterSIM reference data (no R).
2. Injects a group trajectory difference as feature-set surgery on the
   per-stage differential indicators, in one of the modes below.
3. Runs the MOTCO trajectory pipeline (integration → design →
   `estimate_difference` → RRPP).
4. Records per-statistic p-values (`delta`, `angle`, `shape`) plus a
   pre-registered combined rule.

**Trajectory modes**

| Mode          | Role                                                                  |
|---------------|-----------------------------------------------------------------------|
| `none`        | Type I baseline (identical groups). Always added by the enumerator.   |
| `translation` | Negative control — constant location offset (no geometry change).     |
| `magnitude`   | Power probe for `delta` (uniformly scales every step).                |
| `orientation` | Power probe for `angle` (global feature permutation = rotation).      |
| `shape`       | Power probe for `shape` (perturbs interior-stage overlaps).           |

Reports cover three views:

- **Type I table** — rejection rates on `none` cells (target ≈ α).
- **Specificity matrix** — off-diagonal rejection rates: each mode ×
  each statistic. Diagonal entries are power; off-diagonals should
  stay near α.
- **Power curves** — per-statistic rejection rate vs `effect_size` per
  mode.

Optional **acceptance targets** (Type I control, power monotonicity,
specificity) are evaluated against Monte Carlo uncertainty and saved
to a non-gating report.

---

## 2. Module layout

```
src/motco/simulations/study/
├── __init__.py        Public API re-exports
├── config.py          StudyConfig dataclasses + load/dump
├── enumerate.py       StudyConfig → SimulationGrid (cells)
├── sharding.py        Partition grid into shards, run_shard()
├── merge.py           Combine shard_*.jsonl → merged.jsonl
├── summary.py         Per-statistic + combined-rule summaries
├── report.py          Specificity matrix, power curves, Type I table, plots
└── targets.py         Acceptance-target evaluation
```

Driver scripts live at the repo root under `scripts/`:

- `scripts/run_study_shard.py` — one shard per invocation (one cluster array task).
- `scripts/motco_study.py` — `merge` + `report` subcommands for post-processing.
- `scripts/motco_study_array.sbatch` — SLURM array template.

Example configs live under `examples/trajectory_power_study/`:

- `smoke.json` — tiny end-to-end smoke (~minutes locally, not paper-grade).

---

## 3. Pipeline

```
config (YAML or JSON)
    │
    ▼  scripts/run_study_shard.py        ← one cluster array task per shard
    │     writes shard_<i>.jsonl (signature-guarded, resumable)
    ▼  scripts/motco_study.py merge      ← shard_*.jsonl → merged.jsonl
    │
    ▼  scripts/motco_study.py report     ← CSVs + PNGs + acceptance report
```

Two orthogonal parallelism layers:

1. **Across shards** — one SLURM array task per shard
   (`$SLURM_ARRAY_TASK_ID`).
2. **Within a shard** — `--n-jobs $SLURM_CPUS_PER_TASK` forwards to
   RRPP's permutation loop via the evaluation params.

---

## 4. Configuration

Configs are YAML or JSON; both go through `load_study_config()`. The
schema lives in `config.py` (`StudyConfig`). Required top-level keys:

| Field              | Purpose                                                                |
|--------------------|------------------------------------------------------------------------|
| `generator`        | Baseline numpy-generator params (sizing, `n_stages`, `p_dmp`, per-omic `delta_*`, perturbation). `seed` is required. |
| `evaluation`       | Integration method, RRPP permutations, `n_jobs`, eval seed.            |
| `trajectory_modes` | Modes enumerated in the power grid. `none` is always added.            |
| `effect_sizes`     | Non-negative effect-size sweep (per mode).                             |
| `axes`             | Optional OFAT axes. Keys must be namespaced `generator.*` or `evaluation.*`. |
| `n_replicates`     | Replicates per cell.                                                   |
| `base_seed`        | Deterministic seed root for replicates.                                |
| `alpha`            | Significance level for rejection rates.                                |
| `acceptance`       | Pre-specified targets: `type_i`, `power`, `specificity`.               |
| `metadata`         | Free-form provenance (name, intent, notes).                            |

Validation enforces:

- `trajectory_modes` ⊆ `{none, translation, magnitude, orientation, shape}`.
- `effect_sizes` are non-negative.
- `axes` keys use a known namespace prefix (`generator.*` or `evaluation.*`).
- `0 < alpha < 1`.

See `examples/trajectory_power_study/smoke.json` for a complete,
minimal example.

### Scaling smoke → paper-grade

The smoke config is meant to finish in minutes; for a real study you
typically want:

| Field                       | Smoke | Paper-grade (typical) |
|-----------------------------|-------|-----------------------|
| `generator.n_samples`       | 60    | 200–600               |
| `evaluation.permutations`   | 49    | 999 or 4999           |
| `n_replicates`              | 8     | 500–1000              |
| `effect_sizes`              | 4 pts | 5–8 pts incl. `0.0`   |

Keep `0.0` as the first effect size — it anchors the within-mode null
and is what the specificity/Type I checks read.

---

## 5. Local smoke run

From the repo root, with the project virtualenv active:

```bash
uv venv && source .venv/bin/activate
uv sync --extra test
```

Run all shards locally:

```bash
for i in 0 1 2 3; do
  python scripts/run_study_shard.py \
    --config examples/trajectory_power_study/smoke.json \
    --out-dir /tmp/motco-smoke \
    --shard-index $i --n-shards 4 \
    --error-policy record
done

python scripts/motco_study.py merge  --out-dir /tmp/motco-smoke
python scripts/motco_study.py report \
    --config examples/trajectory_power_study/smoke.json \
    --out-dir /tmp/motco-smoke
```

Outputs (under `/tmp/motco-smoke/report/`):

- `specificity_matrix.csv` / `.png`
- `power_curves.csv` / `.png`
- `type_i_table.csv` / `type_i.png`
- `acceptance_report.csv` / `.json`

Run the smoke first on any new environment — it doubles as a sanity
check that the package (and its cached reference data) is wired up
correctly. No R is required at runtime.

---

## 6. SLURM cluster run

### 6.1 Prerequisites on the cluster

- Python (matching `pyproject.toml`) and `uv` (or `pip`) available on
  the compute nodes.
- **No R needed.** Generation runs on the numpy generator and the
  cached reference data (`src/motco/simulations/data/intersim_reference.npz`),
  which ships in the repo. R is only ever needed to *regenerate* that
  cache (see `export_reference.R`), not to run the study.
- The repo cloned and the virtualenv built once on a login node:
  `uv venv && uv sync --extra test`. The sbatch template activates
  `.venv/bin/activate` from the project root.

If your cluster uses `module load`/conda, add the appropriate `module
load python` (and any `conda activate`) lines before the `source
.venv/bin/activate` step in the sbatch script. No R module is required.

### 6.2 Choose `N_SHARDS`

Total replicates = `n_replicates × n_cells`, where `n_cells` is the
sum of (modes × effect_sizes) for the power grid plus one cell per
OFAT axis value. Aim for **~2–6 h per shard**:

```
N_SHARDS ≈ ceil(total_replicate_seconds / target_shard_seconds)
```

A short probe submission (`--array=0-0 --time=2:00:00`) gives a
realistic per-shard wallclock and memory profile. Use that to pick
both `N_SHARDS` and `--cpus-per-task`.

### 6.3 Edit the sbatch template

Open `scripts/motco_study_array.sbatch` and adjust:

- `#SBATCH --time`, `--cpus-per-task`, `--mem` — set from your probe.
- Partition / QOS / account flags (`-p`, `--qos`, `-A`) — cluster-specific.
- The `#SBATCH --array=0-N_SHARDS_MINUS_ONE` line is a placeholder;
  override with `--array=` on the command line at submit time.
- If your cluster needs `module load` / conda, add those calls before
  the `.venv/bin/activate` line.

The runner forwards `--n-jobs $SLURM_CPUS_PER_TASK` so RRPP saturates
the CPUs allocated to each task.

### 6.4 Submit

```bash
mkdir -p logs results

sbatch \
  --array=0-63 \
  --export=ALL,STUDY_CONFIG=$(pwd)/examples/trajectory_power_study/study.json,STUDY_OUT=$(pwd)/results,N_SHARDS=64 \
  scripts/motco_study_array.sbatch
```

Required environment variables (via `--export`):

| Variable       | Meaning                                                         |
|----------------|-----------------------------------------------------------------|
| `STUDY_CONFIG` | Absolute path to the study config (YAML or JSON).               |
| `STUDY_OUT`    | Output directory for `shard_<i>.jsonl` files.                   |
| `N_SHARDS`     | Total shards. Must match the `--array` width.                   |

### 6.5 Merge + report after completion

```bash
python scripts/motco_study.py merge  --out-dir results
python scripts/motco_study.py report \
    --config examples/trajectory_power_study/study.json \
    --out-dir results
```

---

## 7. Failure handling and resumption

- **`--error-policy record`** (default in the sbatch template) lets a
  shard continue past a failed replicate; the failure is captured as
  a row with `status="failed"` in the shard JSONL. Use `raise` only
  when you want a single failure to abort the shard.
- **Signature-guarded resumability** — each `shard_<i>.jsonl` records
  the parameter signature it was produced with. Re-running the same
  shard skips replicates already completed for that signature and
  fills only what's missing or failed.
- **Resubmit only failed array tasks**:

  ```bash
  sbatch --array=7,12,40 \
         --export=ALL,STUDY_CONFIG=...,STUDY_OUT=...,N_SHARDS=64 \
         scripts/motco_study_array.sbatch
  ```

- **Forced overwrite** — pass `--overwrite` to `run_study_shard.py` to
  discard an existing shard JSONL before running. Use with care; this
  loses any completed replicates in that shard.

---

## 8. Outputs

After `merge` + `report`, the output directory looks like:

```
results/
├── shard_0.jsonl            (per-shard raw records, signature-guarded)
├── shard_1.jsonl
├── ...
├── merged.jsonl             (deduplicated by (cell, replicate))
└── report/
    ├── specificity_matrix.csv     mode × statistic rejection rates
    ├── specificity_matrix.png
    ├── power_curves.csv           rejection rate vs effect_size, per mode × statistic
    ├── power_curves.png
    ├── type_i_table.csv           per-statistic + combined-rule on null cells
    ├── type_i.png
    ├── acceptance_report.csv      acceptance target evaluation
    └── acceptance_report.json
```

Interpretation notes:

- **Diagonal of the specificity matrix** is power at the largest
  `effect_size` for the matching statistic; off-diagonals are
  specificity (should sit near α).
- **Type I table** reads the `none` cells. Both per-statistic and the
  combined rule should respect α within Monte Carlo uncertainty.
- **Acceptance report** is non-gating — it's a structured record of
  whether pre-specified targets were met given the SE of the
  rejection-rate estimates.

---

## 9. Programmatic API

The CLI scripts are thin wrappers around the public API exposed in
`motco.simulations.study.__init__`. You can drive everything from
Python if you prefer:

```python
from pathlib import Path
from motco.simulations.study import (
    load_study_config, enumerate_study, run_shard,
    summarize_study, summarize_combined_rule,
    build_specificity_matrix, build_power_curves, build_type_i_table,
    evaluate_targets, write_report_csvs, write_target_report,
)

config = load_study_config("examples/trajectory_power_study/smoke.json")
grid   = enumerate_study(config)

# Shard 0 of 4, in-process:
records = run_shard(grid, shard_index=0, n_shards=4,
                    out_dir=Path("/tmp/motco-smoke"),
                    error_policy="record")
```

---

## 10. Troubleshooting

- **Reference cache missing** (`ReferenceCacheMissingError`) — the
  committed `data/intersim_reference.npz` is absent from the install.
  Reinstall the package, or regenerate it once in an R environment with
  InterSIM: `Rscript src/motco/simulations/export_reference.R
  --output-dir <dir>` then `build_cache_from_export(<dir>)`. No R is
  needed for normal runs.
- **Shard wallclock spills past `#SBATCH --time`** — the shard is
  resumable: just resubmit the failed array task ids. Then either
  lower the per-shard load (raise `N_SHARDS`) or raise `--time`.
- **Memory blowups under high `n_jobs`** — RRPP's parallel workers
  each hold a copy of the permuted residuals. Lower
  `--cpus-per-task` and re-probe.
- **Reports disagree with what you expect on a null cell** — check
  that `effect_sizes` starts at `0.0` and that `none` survived
  enumeration; both are required for the Type I view.
- **Re-running an old shard skips everything / produces no new
  records** — the parameter signature includes a seed-derivation
  version. After changes to derivation logic or the generator surface
  (e.g. the move to the numpy generator), shard files produced before
  the change have a stale signature and are re-executed automatically
  on the next `run_shard`. No manual deletion needed.
