## Why

The semi-synthetic studies currently compare a requested trajectory mode with statistics measured after integration, without recording the geometry actually created before integration. This prevents generator construction, finite-sample variation, preprocessing, and PLS projection from being distinguished when orientation or shape produces off-target responses.

## What Changes

- Add a full realized-geometry decomposition for semi-synthetic trajectory datasets at four checkpoints: native population means, standardized population means, observed standardized features, and PLS latent scores.
- Report `delta`, `angle`, and, when available, `shape`, together with each group's path length, at every applicable checkpoint.
- Report population and observed diagnostics separately for methylation, expression, and proteomics, plus joint diagnostics after the omic blocks share a meaningful standardized scale.
- Use methylation M-values and the same pooled, per-feature standardization parameters for the population and observed pre-integration checkpoints.
- Persist structured construction-level diagnostics with each study replicate so mode-by-statistic results can be attributed to construction, sampling, preprocessing, or projection.
- Characterize orientation and shape across requested effect sizes, including the effects propagated through the CpG-to-gene-to-protein cascade, and document achievable construction contracts rather than assuming perfect statistic specificity.

## Capabilities

### New Capabilities

- `realized-geometry-diagnostics`: Decompose requested semi-synthetic trajectory effects into population, preprocessing, sampling, and PLS-latent geometry with per-omic and joint structured diagnostics.

### Modified Capabilities

None.

## Impact

- Affects the semi-synthetic generator truth representation, evaluation preprocessing boundary, trajectory evaluation result schema, grid/JSONL persistence, and study reporting utilities under `src/motco/simulations/`.
- Adds deterministic and numerical regression coverage under `tests/` for analytic means, shared preprocessing, checkpoint geometry, serialization, and orientation/shape behavior.
- Extends study outputs with additive diagnostic fields; existing trajectory statistics and public analysis semantics remain unchanged.
- Informs the Phase 2 construction contract and gates the subsequent orientation-driver and medium-pilot work documented in `docs/roadmap.md`.
