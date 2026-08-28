## Why

MOTCO's July medium pilot predates the corrected shape estimator, realized-geometry diagnostics, and production orientation-driver attribution. A new Phase 4 pilot is required before the paper-grade study so operating characteristics are interpreted against the geometry actually generated and the stability of the fitted PLS representation and its feature attribution is measured rather than assumed.

## What Changes

- Add a versioned Phase 4 medium-pilot configuration using pooled PLS on M-value methylation, 300 samples, four stages, 100 replicates per cell, 199 permutations, matched seeds with a single shared zero-effect anchor cell, the five-effect grid, and the established magnitude, orientation, shape, and translation modes.
- Preserve selected PLS component and integration metadata in each persisted replicate record.
- Add an optional, bounded orientation-attribution diagnostic that reuses the exact fitted PLS model and pooled standardized feature matrix from the trajectory evaluation, runs only for configured orientation cells, and persists compact transition, top-feature, bootstrap-stability, and generator-truth recovery summaries.
- Extend study reporting to summarize all realized-geometry checkpoints beside rejection rates, characterize selected-component and attribution stability, and distinguish expected construction co-movement from estimator artifacts.
- Replace blanket off-diagonal specificity judgments with predeclared Phase 4 gates whose parameters are declared in the committed config and consumed by the study code: Type I control under `none` and every translation effect level, magnitude specificity, orientation power and monotonicity, shape behavior under the corrected invariance contract, complete diagnostics, a confirmation rule so one marginal control cannot decide the phase, and an explicit proceed/hold decision for Phase 5.
- Produce a dated, versioned findings report with exact reproduction commands and retain the July pilot as historical evidence rather than overwriting it.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `simulation-evaluation-harness`: Optionally retain the fitted pooled PLS analysis boundary long enough to compute compact orientation-attribution diagnostics and return them with integration metadata.
- `trajectory-power-study`: Persist PLS and attribution diagnostics, provide the fixed Phase 4 pilot profile, report operating characteristics against realized geometry, and emit a Phase 5 gate decision.

## Impact

- Affects PLS integration and evaluation result handling under `src/motco/simulations/evaluation.py`, while preserving existing trajectory statistics and public defaults.
- Extends simulation replicate persistence, parameter signatures, study configuration, summaries, and reports under `src/motco/simulations/grid.py` and `src/motco/simulations/study/`.
- Adds a new Phase 4 config under `examples/trajectory_power_study/` and a findings report under `docs/reports/`; existing configs and reports remain unchanged.
- Updates `docs/roadmap.md` with the Phase 4 gate decision and the study README and API documentation with the new config fields, diagnostic semantics, and commands.
- Reuses `motco.stats.attribution` without changing its public contract and does not change InterSIM invocation, the numpy generator, RRPP, trajectory estimators, or SNF behavior.
- Adds no external runtime dependency; the study remains executable from cached reference data without R.
