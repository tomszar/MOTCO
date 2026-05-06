## Why

MOTCO needs a reproducible way to generate semi-synthetic multi-omics inputs before we can evaluate power and Type I error for trajectory tests. InterSIM already simulates methylation, gene expression, and protein expression from real TCGA-derived covariance and cross-omic structure, but MOTCO currently has no Python contract for invoking it or consuming its outputs.

## What Changes

- Add a Python-facing InterSIM bridge that invokes the R `InterSIM` package from MOTCO development tooling
- Return InterSIM outputs as stable Python objects containing methylation, gene expression, protein expression, sample IDs, cluster assignments, and simulation metadata
- Validate R and InterSIM availability with clear errors when the dependency is missing
- Support deterministic simulation through explicit seed handling
- Provide smoke tests that prove Python can invoke InterSIM and receive aligned omics matrices plus clustering metadata
- Keep this change limited to InterSIM invocation and output normalization; trajectory injection, simulation grids, power estimation, and Type I error analysis remain out of scope

## Capabilities

### New Capabilities

- `intersim-simulation-bridge`: Python bridge for invoking R InterSIM and returning aligned methylation, gene expression, protein expression, cluster labels, sample IDs, and metadata

### Modified Capabilities

*(none - no existing spec-level requirements are changing)*

## Impact

- **New code area**: a simulation support module under the MOTCO package namespace, replacing or superseding the current `src/simulations` mockup boundary as needed
- **New optional external dependency**: R with the CRAN/R-universe `InterSIM` package installed
- **Potential Python dependency**: either `rpy2` as an optional bridge dependency or a subprocess-based `Rscript` invocation path; the design will choose the first implementation path
- **Tests**: dependency-aware smoke tests that skip when R or InterSIM is unavailable, plus unit tests for output normalization and validation
- **No breaking changes** to existing MOTCO statistical APIs or CLI commands
