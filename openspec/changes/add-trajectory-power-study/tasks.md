## 1. Study configuration

- [x] 1.1 Create `src/motco/simulations/study/` package with `__init__.py` re-exporting the public surface
- [x] 1.2 Define the config schema (dataclasses) for baseline `intersim`/`generator`/`evaluation` params, `trajectory_modes`, `effect_sizes`, OFAT `axes`, `n_replicates`, `base_seed`, and an `acceptance` block
- [x] 1.3 Implement a config loader (YAML, JSON fallback) with validation: required fields, known trajectory modes, non-negative replicate count, namespaced axes (`intersim.`/`generator.`/`evaluation.`)
- [x] 1.4 Implement `enumerate_study(config)` that builds the combined `SimulationGrid` via `enumerate_type_i_grid` + `enumerate_power_grid`, ensuring `none` and `translation` appear as Type I negative controls
- [x] 1.5 Tests: deterministic/stable cell ids on re-enumeration; negative-control modes present; invalid configs rejected with clear errors

## 2. Sharded execution

- [x] 2.1 Implement deterministic `(cell_id, replicate_index)` unit enumeration and a partition function mapping each unit to a shard in `[0, n_shards)`
- [x] 2.2 Implement a shard runner that filters units to a given `shard_index`, executes them via the existing per-replicate runner, and writes `shard_<i>.jsonl`
- [x] 2.3 Wire resume: re-running a shard skips completed replicates with matching parameter signatures and never duplicates
- [x] 2.4 Honor the configured error policy (record failed replicate + continue, or raise)
- [x] 2.5 Tests: partition is exhaustive and non-overlapping across shards; resume skips completed work; failure recorded under record policy

## 3. Cluster submission

- [x] 3.1 Add a SLURM array template under `scripts/` that submits an array of size `n_shards` and invokes the runner with `--shard-index $SLURM_ARRAY_TASK_ID --n-shards N --config <path> --out-dir <dir>`
- [x] 3.2 Document CPUs-per-task → RRPP `n_jobs` mapping and resubmission of failed task ids in the script header

## 4. Merge

- [x] 4.1 Implement merge that reads all `shard_*.jsonl` via `read_replicate_results`, deduplicates by `(cell_id, replicate_index)`
- [x] 4.2 Raise a clear error when the same key appears with differing parameter signatures
- [x] 4.3 Tests: merge yields one record per key; inconsistent-signature merge raises

## 5. Summarization

- [x] 5.1 Reuse `summarize_rejection_rates` for per-statistic (`delta`/`angle`/`shape`) rates with Monte Carlo SE at configured alpha
- [x] 5.2 Implement the combined-rule Type I statistic over null cells (reject if any available statistic `< alpha`) with Monte Carlo SE
- [x] 5.3 Tests: per-statistic unavailable handling (e.g. `shape` with <3 stages); combined-rule rate on a known synthetic null set

## 6. Reporting

- [x] 6.1 Build the mode × statistic specificity matrix (rate ± SE) by joining summaries on cell metadata; write CSV
- [x] 6.2 Build power-curve frames (rate vs `effect_size` per mode × statistic); write CSV
- [x] 6.3 Build the Type I table (null cells across axes, per-statistic + combined-rule); write CSV
- [x] 6.4 Render figures with matplotlib: specificity matrix heatmap, Type I plot vs alpha, power-curve panel grid with error bars
- [x] 6.5 Tests: matrix/curve/table frames have expected shape and columns from a synthetic summary set

## 7. Acceptance targets

- [x] 7.1 Implement target evaluation: Type I within k·SE of alpha; power monotone in effect size and ≥ floor at top effect; off-diagonal specificity within k·SE of alpha
- [x] 7.2 Emit a per-target met/not-met report (CSV/JSON) carrying the relevant Monte Carlo uncertainty; non-gating
- [x] 7.3 Tests: each target type evaluated correctly on synthetic summaries (met and not-met cases)

## 8. Driver entry points

- [x] 8.1 Add the runner entry point (config + shard index) — standalone script under `scripts/` for cluster use
- [x] 8.2 Add a thin `merge`/`report` entry point (CLI subcommand or script) for interactive post-processing
- [x] 8.3 End-to-end smoke test: tiny config (few cells, few replicates, low permutations) runs runner → merge → report with mocked or minimal InterSIM, producing all CSVs

## 9. Docs & gate

- [x] 9.1 Add a small in-repo smoke config and document the full study workflow (config → cluster array → merge → report) in README/docs
- [x] 9.2 Run the pre-commit gate: `ruff check src/ tests/` + `mypy src/motco/` + fast pytest all pass

