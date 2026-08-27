## Why

MOTCO can detect a group orientation difference in its pooled PLS molecular space, but it cannot yet explain which molecular features account for the difference or how much of the observed contrast is retained by PLS. Phase 3 addresses this interpretation gap before the next operating-characteristic pilot, while preserving the single pooled coordinate system required by the production analysis.

## What Changes

- Add a production orientation-attribution capability for two-group, multi-stage data.
- Compute observed group-by-transition feature contrasts from pooled-standardized data or covariate-adjusted LS means, normalized to separate directional change from path length.
- Reconstruct the same group-stage transitions through one frozen fitted PLS model and report the PLS-captured contrast and observed-minus-captured residual.
- Support adjacent transitions for trajectories with more than two stages and retain signed feature-level contributions.
- Add bootstrap resampling within group and stage strata to estimate feature sign stability, rank stability, and top-k selection frequency.
- Provide optional correlated-feature module and caller-supplied pathway aggregation without adding a pathway database dependency.
- Return typed, machine-readable attribution results and provide findings-ready tabular output; document standardized versus original-unit effects and interpretation limits.

## Capabilities

### New Capabilities

- `orientation-driver-attribution`: Feature-, module-, and pathway-level attribution of significant group trajectory orientation differences in a shared PLS space, including PLS reconstruction, residuals, and bootstrap stability.

### Modified Capabilities

## Impact

- Adds a new attribution module under `src/motco/` and public exports for its result and analysis functions.
- Reuses `FittedOmicsPreprocessor`, `fit_plsda_model`, PLS estimator transforms, design/LS-means utilities, and existing trajectory conventions.
- Adds unit and integration tests for normalization, reconstruction, residual accounting, multi-stage transitions, bootstrap determinism, and module/pathway aggregation.
- Extends simulation and analysis documentation with the attribution contract and its non-causal interpretation boundary.
- Does not change InterSIM invocation, RRPP p-values, the existing trajectory estimators, pooled VIP semantics, or the PLS latent-space selection procedure.
