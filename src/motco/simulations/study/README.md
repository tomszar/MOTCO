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
| `design_grid`      | Optional **crossed** axes (`{"axes": {...}}`); every combination is a design point with its own anchored power grid. Each axis must list its baseline value. |
| `n_replicates`     | Replicates per cell.                                                   |
| `base_seed`        | Deterministic seed root for replicates.                                |
| `alpha`            | Significance level for rejection rates.                                |
| `acceptance`       | Pre-specified targets: `type_i`, `power`, `specificity`, optional `gate`, optional `design_point` (advisory design-point rule over a `design_grid`). |
| `metadata`         | Free-form provenance (name, intent, notes).                            |

Validation enforces:

- `trajectory_modes` ⊆ `{none, translation, magnitude, orientation, shape}`.
- `effect_sizes` are non-negative.
- `axes` keys use a known namespace prefix (`generator.*` or `evaluation.*`).
- `design_grid.axes` keys use the same namespaces, are not also OFAT axes, and
  each lists the baseline value; `acceptance.design_point.prefer` names only
  design-grid axes.
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
    ├── config_spectrum.csv        recorded latent eigengaps per cell × configuration
    ├── eigengap_stratified_power.csv  orientation power by eigengap tercile
    ├── continuity_resolved_orientation.csv  only when the grid sweeps baseline continuity
    ├── design_point_operating.csv only when the config declares a design_grid
    ├── design_point_power.png     (same condition)
    ├── design_point_decision.json only when acceptance.design_point is declared
    ├── design_point_decision.csv  (same condition)
    ├── acceptance_report.csv      acceptance target evaluation
    └── acceptance_report.json
```

### Per-replicate record fields

Every JSONL record is one `SimulationReplicateResult`. Beyond the cell identity,
seeds, `p_values`, and `pair_statistics`, two fields carry the per-replicate
diagnostics:

- `realized_geometry` / `integration_metadata` / `attribution_diagnostics` — the
  Phase 4 diagnostics.
- `null_summary` — a compact per-statistic description of *that replicate's own*
  RRPP permutation null: `count` (retained draws, non-finite excluded), `mean`,
  `sd`, and `q50`/`q90`/`q95`/`q99`. It is written on every run with
  `permutations > 0`, independently of `include_null_distributions` (which
  controls only whether the full draw vectors are *returned*, never persisted).

  `null_summary` makes the pair *(observed statistic, its own critical value)*
  computable from a single record, which is what any pivotality or calibration
  question needs. It is **not** part of `parameter_signature` — it summarizes
  draws that already happened and changes no generation, integration, or
  permutation behavior — so records written before it existed still load (with an
  empty summary) and remain resumable. A record from a `permutations = 0` run is
  told apart from a pre-field record by `runtime_metadata["permutations"]`.

- `config_spectrum` — the eigenspectrum of the centered **stage-mean
  configuration in the evaluated latent space**, written on every run. It carries
  `pooled` (stages averaged across groups) and `groups` (one entry per group
  level), each with `n_points`, `n_dimensions`, `total_variance`, the normalized
  `spectrum`, and the **relative eigengap** `(l1 − l2) / Σl`. With
  `permutations > 0` it also carries `permutation_pooled_eigengap` — `count`,
  `mean`, `sd`, `q05`/`q50`/`q95` of the pooled eigengap over the permutation
  draws — so a replicate can be located against its own permutation
  distribution. Full per-permutation spectra are never retained.

  The eigengap is the covariate the [geometry
  audit](../../../../docs/reports/geometry-audit-2026-09-01.md) found predicts how
  wide a replicate's own `angle` permutation null will be, and therefore whether
  its orientation is resolvable at all. It is **recorded and reported only** — it
  enters no statistic, no p-value, and no decision rule. A configuration with
  zero total variance records `relative_eigengap: null` and an empty `spectrum`
  rather than a non-finite float; a two-stage configuration's eigengap is
  identically `1.0` and uninformative by construction.

  Unlike `null_summary`, the spectrum **is** signature-bearing:
  `parameter_signature` includes `config_spectrum_version`. A shard written
  before this field existed therefore **refuses resume** with the usual
  signature-mismatch error, rather than producing a merged set in which only some
  records carry the covariate — a missing covariate and a degenerate one are
  indistinguishable at analysis time. Already-merged pre-change record sets still
  *load* (the block is empty) and their committed reports are unaffected; only
  resuming *into* an old shard is refused.

### Eigengap reporting

`motco_study.py report` writes up to three spectrum tables beside the existing
ones, from recorded values only — no dataset is regenerated and no spectrum
recomputed:

- `config_spectrum.csv` — one row per (cell, configuration ∈ {`pooled`, group
  levels}) with `n_replicates` / `n_recorded` / `n_available` and the eigengap
  mean, sd, quartiles, min and max. `n_recorded` counts records carrying the
  block at all and `n_available` those whose eigengap is defined, so pre-field
  records and degenerate configurations stay separately visible.
- `eigengap_stratified_power.csv` — for orientation-mode power cells, rejection
  rates within **within-cell terciles** of the recorded pooled eigengap, with
  per-stratum counts and Monte Carlo SEs. A cell whose records carry no spectrum
  is emitted with `status = unavailable` rather than dropped.
- `continuity_resolved_orientation.csv` — **only when the merged set spans more
  than one `generator.baseline_continuity` value.** One row per (continuity,
  mode, effect, statistic) with the rejection rate, the recorded pooled-eigengap
  distribution (mean and terciles), and the dispersion of
  `null_summary["angle"]["q95"]`. The eigengap is the point of the table: power
  that rises along ρ should be traceable to a configuration that acquired a
  dominant axis, which is the observable that carries a continuity-conditioned
  orientation claim to real data — not the knob itself. A study that holds the
  axis fixed writes no such file, and records predating the axis are excluded
  rather than folded into the ρ = 0 bin. When the grid is a `design_grid`, rows
  are additionally keyed on every other design coordinate (emitted as columns),
  so a ρ × `n_samples` grid never pools three operating points into one row.

### Design-point reporting

A config with a `design_grid` enumerates one anchored power grid per design
point (phase `power_design`, metadata `design_point = {axis: value}`,
`varied_axis = "design_grid"`); the baseline point is the primary grid, stamped
with its coordinates. Design cells join the primary matched-seed family and are
invisible to every baseline reader (power curves, specificity matrix, Type I
table, Phase 4 gate, acceptance targets). The report then writes:

- `design_point_operating.csv` — one row per (design coordinates, mode, effect,
  statistic), anchors included as `none` at `0.0`: rejection rate ± MC SE, the
  recorded pooled-eigengap distribution, the `angle` null-width (`q95`)
  dispersion, and the selected-dimensionality distribution.
- `design_point_power.png` — the target statistic's power at the top effect per
  point against the first design axis, one line per value of the second,
  annotated with the median eigengap.
- `design_point_decision.json` / `.csv` — when `acceptance.design_point`
  declares the rule (`trajectory_mode`, `statistic`, `min_power_at_top`,
  `confirmation_se_threshold`, `prefer`): per column `meets` (`rate − k·SE ≥
  floor`), `marginal` (point estimate only), `fails`, or `unavailable`, with
  the anchor's Type I rates beside it, and a verdict — the first `meets` column
  in preference order, or `revise_claim`. Advisory only.

The committed Phase 5 design-point pilot,
`examples/trajectory_power_study/phase5_design_point_pilot.json`, crosses
ρ ∈ {0.0, 0.5, 0.8} with `n_samples` ∈ {300, 600, 1200} at `p_dmp = 0.1`.

### Angle-pivotality diagnostic

`scripts/angle_null_pivotality.py` reads a merged record set and reports whether
each replicate's null moved with its own observed statistic:

```bash
# 5 cells x 100 replicates = 500 work units; ~26 minutes on 16 cores.
for i in $(seq 0 15); do
  python scripts/run_study_shard.py \
      --config examples/trajectory_power_study/angle_pivotality_diagnostic.json \
      --out-dir results/angle-pivotality-2026-09-01 \
      --shard-index "$i" --n-shards 16 --error-policy record &
done
wait
python scripts/motco_study.py merge --out-dir results/angle-pivotality-2026-09-01
python scripts/angle_null_pivotality.py \
    --merged  results/angle-pivotality-2026-09-01/merged.jsonl \
    --out-dir results/angle-pivotality-2026-09-01/report
```

Do **not** pass `--n-jobs`: it is part of the cell parameter signature, so
overriding it changes the permutation draws and breaks resume.

It writes `pivotality_association.csv` (correlation and slope of the null's
mean/sd/q95 against the observed statistic, with a Fisher-z interval),
`pivotality_rejection_split.csv` (observed statistic and critical value split by
rejection outcome), `pivotality_standardized.csv` (as-specified vs
cross-replicate standardized rejection rate, controls included), and
`pivotality_spectrum.csv` (per cell and statistic, the Spearman and log–log
Pearson association between the *recorded* pooled eigengap and the width of that
replicate's own null, with Fisher-z intervals; `status = unavailable` for record
sets predating the spectrum field).

The standardized counterfactual is a **diagnostic, not a deployable test** — it
borrows a reference `z` distribution from the null-control cells. Within-replicate
studentization is a no-op: it rescales both sides of the comparison and leaves
the p-value unchanged.

Note that `enumerate_study` adds two cells the profile does not request — a
`none` Type I baseline and a `translation` negative control. They are kept on
purpose: they are the null-control reference the standardized counterfactual
calibrates against.

The 2026-09-01 run of this profile is written up in
`docs/reports/angle-null-pivotality-2026-09-01.md`.

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
