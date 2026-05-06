## Why

MOTCO can now generate known-truth semi-synthetic trajectory datasets, but there is no packaged harness that runs those datasets through MOTCO integration and trajectory testing. Before estimating power or Type I error across large grids, MOTCO needs a reproducible per-replicate evaluation layer that converts generated datasets into observed statistics, RRPP null distributions, and p-values.

## What Changes

- Add a simulation evaluation harness that consumes `SemiSyntheticTrajectoryDataset`
- Provide an initial integration path for MOTCO-supported latent spaces, starting with SNF spectral embedding and/or a simple concatenated feature matrix baseline
- Build MOTCO model matrices, LS means, and trajectory contrasts from generated sample metadata
- Run observed trajectory difference estimation for `delta`, `angle`, and `shape`
- Optionally run RRPP permutation testing and compute p-values for the observed statistics
- Return a structured per-replicate result containing observed statistics, p-values, runtime metadata, generator truth metadata, and evaluation parameters
- Keep this change focused on evaluating one generated dataset or one replicate; grid enumeration, batch execution, checkpointing, and report generation remain out of scope

## Capabilities

### New Capabilities

- `simulation-evaluation-harness`: Evaluate one semi-synthetic trajectory dataset through MOTCO integration and trajectory testing, returning observed statistics, optional RRPP p-values, and metadata

### Modified Capabilities

*(none - existing stats and simulation-generation contracts are consumed but not changed)*

## Impact

- **New code area**: evaluation harness module under `src/motco/simulations/`
- **Depends on**: `semisynthetic-trajectory-generator`, `motco.stats.design`, `motco.stats.trajectory`, `motco.stats.permutation`, and selected integration functions
- **API additions**: evaluation parameter/result models plus functions for evaluating one dataset or one replicate
- **Tests**: deterministic unit tests using small generated fixtures and low permutation counts
- **Documentation**: describe evaluation flow, supported integration methods, output result schema, and limitations
- **No breaking changes** to existing stats, InterSIM bridge, generator, or CLI APIs
