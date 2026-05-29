## Context

The `simulation-grid-orchestration` engine (`src/motco/simulations/grid.py`) already enumerates Type I and power grids, runs deterministic replicates through the evaluation harness, persists resumable JSONL, and computes per-statistic rejection summaries with Monte Carlo SE. It is, however, library-only and serial: there is no declarative experiment record, no parallel execution model, and no reporting. The trajectory test emits three statistics — `delta` (size), `angle` (orientation), `shape` — tested via RRPP on the group×stage interaction.

Two facts about the generator shape the design:
- Under `trajectory_mode="none"`, groups are assigned at random within stages, so the two groups are an exchangeable random partition — a clean Type I null.
- Under `trajectory_mode="translation"`, group B is shifted by the **same** vector at every stage (coefficients `np.ones`), so trajectory geometry (size/orientation/shape) is unchanged. Translation is therefore a *second* null: a real group effect the trajectory test is correctly blind to.

This makes the natural deliverable a **mode × statistic specificity matrix**: the diagonal is power, the off-diagonal and the `none`/`translation` rows are Type I / specificity.

## Goals / Non-Goals

**Goals:**
- A declarative, version-controlled study config that fully determines the enumerated grid and the acceptance targets.
- An embarrassingly-parallel, resumable execution model that maps cleanly onto a SLURM array, reusing the existing parameter-signature resume guard.
- A merge + reporting layer that produces the specificity matrix, Type I tables, and power-curve data as both CSV and figures, plus an evaluation of pre-specified targets.
- Per-statistic characterization as the primary analysis; combined-rule Type I as a secondary result.

**Non-Goals:**
- No changes to the trajectory test, RRPP, integration methods, or the generator. The study *characterizes* existing behavior; it does not tune it.
- No new general-purpose job scheduler — SLURM array + per-shard files is sufficient; we do not build a distributed coordinator.
- No multiplicity correction on the primary analysis (per-statistic marginals are reported directly).
- No live database — JSONL shards + merge remain the persistence model.

## Decisions

### D1: One new capability, four cohesive modules

Add the study as a single capability `trajectory-power-study`, implemented as new modules under `src/motco/simulations/study/` (config, run/shard, merge, report). Rationale: the four pieces only make sense together as one workflow, and they consume — but do not modify — `grid.py`. Alternative considered: extend `grid.py` directly. Rejected: it would bloat the engine and entangle the experiment record with the orchestration primitives.

### D2: Config is declarative data (YAML), deserialized into existing param dataclasses

The config file lists baseline `intersim`/`generator`/`evaluation` params, `trajectory_modes`, `effect_sizes`, OFAT `axes` (namespaced `intersim.*` / `generator.*` / `evaluation.*`, matching `_split_axis`), `n_replicates`, `base_seed`, and an `acceptance` block. A loader validates and feeds `enumerate_type_i_grid` + `enumerate_power_grid`, concatenating their cells into one `SimulationGrid`. Rationale: the config becomes the paper's provenance artifact; enumeration logic is already built and tested. Alternative: argparse flags. Rejected: a multi-axis grid is unwieldy and unreproducible on the command line. YAML chosen over JSON for human-editing comfort (a small parser dep, or reuse one already present).

### D3: Sharding by deterministic partition of `(cell, replicate)` units

Enumerate all `(cell_id, replicate_index)` units, sort them deterministically, and assign unit `u` to shard `hash(cell_id, replicate_index) % n_shards` (or contiguous block partition). Each shard writes `shard_<i>.jsonl`. A thin wrapper over the existing per-replicate runner filters to the shard's units; the existing parameter-signature guard provides resume. Rationale: independent files avoid concurrent-append races on a shared filesystem; deterministic partition makes a shard's workload reproducible and lets a failed array task simply re-run. Alternative: single shared JSONL with file locking. Rejected: fragile on cluster filesystems. Alternative: one file per cell. Rejected: thousands of tiny files; sharding by a fixed `n_shards` matches SLURM array width.

### D4: SLURM array script reads `n_shards` from array width, shard index from `$SLURM_ARRAY_TASK_ID`

A template script under `scripts/` submits an array of size `n_shards`; each task invokes the runner with `--shard-index $SLURM_ARRAY_TASK_ID --n-shards <N> --config <path> --out-dir <dir>`. RRPP keeps its internal `n_jobs` for within-replicate parallelism (CPUs per task). Rationale: two orthogonal parallelism layers — cells across array tasks, permutations within a task — both already supported.

### D5: Merge validates signature consistency, then reuses `summarize_rejection_rates`

Merge reads all `shard_*.jsonl` via the existing `read_replicate_results`, deduplicates by `(cell_id, replicate_index)`, and raises on a signature mismatch for the same key. The deduplicated list feeds the existing `summarize_rejection_rates` (per-statistic). The combined-rule Type I is computed separately over null cells: a replicate rejects if any available statistic's p-value `< alpha`. Rationale: maximal reuse; the only genuinely new statistic is the combined rule.

### D6: Reporting pivots cell metadata into matrix / curves; figures via matplotlib

Cells already carry `trajectory_mode` and `effect_size` in `metadata`. Reporting joins summaries on that metadata to build: (a) the specificity matrix (mode × statistic, rate ± SE), (b) Type I tables (null cells × axes), (c) power-curve frames (rate vs effect_size per mode × statistic). CSVs are the primary, citable outputs; figures use matplotlib (already a dependency via `viz.py`). Target evaluation compares each summary against the config's `acceptance` block and emits a per-target met/not-met verdict with the relevant Monte Carlo uncertainty.

### D7: Acceptance targets are declared, evaluated, but non-gating

Targets live in the config and are *reported* (Type I within k·SE of alpha; power monotone and ≥ floor at top effect; off-diagonal within k·SE of alpha). For a paper they substantiate claims rather than gate CI. Rationale: a methods paper reports operating characteristics with uncertainty; pre-specifying targets prevents post-hoc storytelling. A small/fast config can still be wired into CI as a smoke check later, but that is out of scope here.

## Risks / Trade-offs

- **Paper-grade replicate counts are expensive** (each replicate spawns an InterSIM R subprocess + RRPP permutations). → Sharding + cluster array makes wall-clock tractable; a separate small "smoke" config validates the pipeline end-to-end before the full run.
- **Per-shard files proliferate / partial cluster failures** leave gaps. → Deterministic partition + signature-guarded resume means re-submitting the array (or specific task ids) backfills exactly the missing units; merge surfaces any remaining gaps via completed-count reporting.
- **Translation may not be a perfect null in finite samples** if the integration step (especially SNF's nonlinear embedding) introduces stage-dependent distortion. → This is itself a study finding worth reporting; the negative-control rows make any such inflation visible rather than hidden.
- **Combined-rule Type I will be inflated by construction** (three correlated tests). → It is reported as a secondary result precisely to quantify that cost for users; the primary per-statistic analysis is unaffected.
- **YAML dependency** if not already present. → Trade-off is minor; JSON is the fallback with no new dependency.

## Open Questions

- Exact axis values and replicate counts (left to the config the user fills in; the design fixes the mechanism, not the numbers).
- Whether the runner is exposed as a `motco` CLI subcommand or a standalone `scripts/` entry point — both are viable; leaning toward a script for the runner (cluster-facing) and a thin CLI for merge/report (interactive). To be finalized in tasks.
- Whether to bundle a small "smoke" config in-repo for CI now or defer.
