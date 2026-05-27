## Why

The simulation grid orchestration engine (`simulation-grid-orchestration`) can enumerate Type I and power cells, run resumable replicates, and summarize rejection rates — but nothing actually *runs* a designed study or turns the output into evidence. As a result, the false-positive rate and power of the MOTCO trajectory test (delta/angle/shape statistics under RRPP) have never been characterized. A methods paper needs that characterization: a reproducible, declaratively-specified study, executed at paper-grade replicate counts on a cluster, reported as a per-statistic specificity matrix with Type I tables and power curves against pre-specified targets.

## What Changes

- Add a **declarative study config** (file-based) that captures baseline parameters, one-factor-at-a-time axes, trajectory modes, effect sizes, replicate counts, and pre-specified acceptance targets — the version-controlled experiment record for the paper.
- Add a **shard-aware runner** that splits the enumerated grid's `(cell, replicate)` units across `N` workers, each writing its own resumable JSONL shard (no concurrent-append races), plus a SLURM array submission script for cluster execution.
- Add a **merge step** that combines per-shard JSONL into a deduplicated result set and feeds the existing `summarize_rejection_rates`.
- Add a **reporting module** that turns summaries into the **mode × statistic specificity matrix**, Type I tables, and power-curve panels (CSV + figures).
- Adopt **per-statistic characterization** (each of delta/angle/shape reported as its own marginal operating curve; no multiplicity correction) as the primary analysis, with a **secondary combined-rule Type I** result reporting the "reject if any statistic is significant" false-positive rate.
- Treat `trajectory_mode="none"` (no group effect) and `trajectory_mode="translation"` (location-only group effect, invisible to trajectory geometry) as **negative controls**: both must reject at the nominal level.

## Capabilities

### New Capabilities
- `trajectory-power-study`: A declaratively-configured, cluster-executable study that characterizes the Type I error and power of the trajectory test via per-statistic operating characteristics, with negative controls, pre-specified acceptance targets, and paper-ready reporting (specificity matrix, Type I tables, power curves).

### Modified Capabilities
<!-- None. The study consumes simulation-grid-orchestration's existing enumeration, replicate execution, persistence, and summarization without changing their requirements. -->

## Impact

- **New module(s)** under `src/motco/simulations/` for the study config, shard-aware runner, merge, and reporting (consuming existing `grid.py` functions).
- **New CLI/driver surface**: a study runner entry point (config + shard index) and a merge/report entry point.
- **New script**: a SLURM array submission template under `scripts/`.
- **New dependency (likely)**: a plotting dependency for power-curve/matrix figures (e.g., matplotlib, already used in `viz.py`); CSV outputs require no new dependency.
- **Reuses** `enumerate_type_i_grid`, `enumerate_power_grid`, `run_simulation_grid` (extended for sharding), `read_replicate_results`, `append_replicate_results`, and `summarize_rejection_rates` from `simulation-grid-orchestration` — no changes to their contracts.
- **No impact** on the core stats modules (`pls`, `snf`, `trajectory`, `permutation`) or the existing `motco simulate` command.
