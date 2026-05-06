## Context

The simulation stack is intentionally layered:

```text
InterSIM bridge
   -> semi-synthetic trajectory generator
      -> simulation evaluation harness
         -> grid orchestration and summaries
```

This change is the top layer. It should manage repeated execution and aggregation, not reimplement data generation, integration, or RRPP. The existing mockup in `src/simulations/motco_simulation_grid.py` is useful conceptual input, but implementation should live under `src/motco/simulations/` and be packaged.

## Goals / Non-Goals

**Goals:**

- Define a typed simulation cell schema
- Enumerate small Type I error and power grids from parameter axes
- Run replicates for each cell through the evaluation harness
- Assign deterministic seeds per cell/replicate
- Persist one row per replicate with cell metadata, truth metadata, observed statistics, p-values, and runtime metadata
- Resume interrupted runs by detecting completed cell/replicate outputs
- Aggregate replicate-level results into rejection rates, Monte Carlo standard errors, and confidence intervals

**Non-Goals:**

- Build a full HPC scheduler integration
- Run massive default grids automatically
- Produce publication-quality figures
- Decide final study parameter values for a manuscript
- Add a public CLI unless explicitly requested
- Change generator or evaluation harness behavior

## Decisions

### Keep orchestration independent of evaluation internals

The runner should accept an evaluation callable or use the default evaluation harness. Tests can pass a fake evaluator, while production calls use the real harness.

Rationale: this makes orchestration tests fast and avoids coupling grid correctness to expensive RRPP runs.

### Model cells and replicates explicitly

Use data models conceptually equivalent to:

```python
SimulationCell(
    cell_id=str,
    phase=str,
    intersim_params=InterSIMParams,
    generator_params=SemiSyntheticTrajectoryParams,
    evaluation_params=SimulationEvaluationParams,
    n_replicates=int,
)
```

Replicates derive seeds from a base seed plus stable cell/replicate identifiers.

### Persist replicate-level rows first

The primary output should be one tabular row per cell/replicate. Summary tables can be derived from those rows.

Rationale: replicate-level records are auditable and allow recomputing summaries without rerunning simulations.

### Prefer CSV or JSONL initially unless parquet dependency is accepted

The mockup mentions parquet, but MOTCO currently does not depend on `pyarrow` or `fastparquet`. To avoid adding a heavy dependency prematurely, the first implementation can use CSV for rectangular summary columns or JSONL for nested metadata.

Alternative: add `pyarrow` and write parquet. This is cleaner for nested/large outputs but should be a deliberate dependency decision.

### Summaries compute rejection rates with uncertainty

For each cell and statistic, summary should include:

- number of completed replicates
- number rejected at alpha
- rejection rate
- Monte Carlo standard error: `sqrt(p * (1 - p) / n)`
- an approximate confidence interval or clearly documented interval method

For null cells, rejection rate estimates Type I error. For alternative cells, rejection rate estimates power.

## Risks / Trade-offs

- **Large grids can become expensive quickly** -> Provide small defaults and require explicit replicate counts; do not auto-run large study plans.
- **Nested result metadata does not fit CSV well** -> Use JSON columns or JSONL initially; revisit parquet if result size becomes painful.
- **Resume logic can silently skip stale outputs** -> Include parameter hash/signature in output rows and validate it before skipping.
- **Random seed collisions can bias replicates** -> Derive seeds deterministically from base seed, cell id, and replicate index; record all seeds.
- **Summary metrics depend on alpha and statistic availability** -> Record alpha in summary metadata and report missing/unavailable statistics explicitly.

## Migration Plan

This is additive. It should be implemented after the simulation evaluation harness exists. Existing mockup files can be referenced but do not need to be moved as part of this proposal.
