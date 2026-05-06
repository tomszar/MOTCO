## Why

Once MOTCO can evaluate one semi-synthetic trajectory dataset, the next need is a reproducible study runner that evaluates many parameter cells and replicates. Power and Type I error estimates require systematic grid enumeration, resumable execution, result persistence, and aggregation across simulation replicates.

## What Changes

- Add simulation grid orchestration for enumerating Type I error and power study cells
- Provide a configurable cell schema covering generator parameters, InterSIM parameters, integration settings, RRPP settings, replicate counts, and seeds
- Run one or more replicates per cell by calling the simulation evaluation harness
- Persist per-replicate results to a durable tabular format suitable for later aggregation
- Support resumable execution by skipping completed cell/replicate outputs
- Aggregate per-replicate p-values into rejection rates, Monte Carlo standard errors, and confidence intervals
- Provide a lightweight local runner first; cluster schedulers, SLURM, and large-scale reporting remain optional future work unless explicitly added later
- Keep this change above the evaluation harness; it should not duplicate generator or MOTCO evaluation logic

## Capabilities

### New Capabilities

- `simulation-grid-orchestration`: Enumerate simulation study cells, run replicates through the evaluation harness, persist results, resume incomplete runs, and summarize Type I error/power metrics

### Modified Capabilities

*(none - the orchestration layer consumes existing generator and evaluation capabilities without changing their contracts)*

## Impact

- **New code area**: orchestration module under `src/motco/simulations/`
- **Depends on**: `semisynthetic-trajectory-generator` and `simulation-evaluation-harness`
- **API additions**: grid/cell parameter models, local runner, result writer/reader, and aggregation helpers
- **Optional dependency consideration**: parquet output may require adding `pyarrow` or using CSV/JSONL initially to avoid new dependencies
- **Tests**: small local grids with mocked or low-cost evaluation functions
- **Documentation**: describe study grid schema, result files, resumability, and summary metrics
- **No breaking changes** to existing stats, InterSIM bridge, generator, evaluation harness, or CLI APIs
