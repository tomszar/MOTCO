## 1. Attribution API and input validation

- [x] 1.1 Add `src/motco/stats/attribution.py` with typed configuration and result records for ordered groups, stages, transitions, feature effects, aggregate effects, bootstrap summaries, and interpretation metadata.
- [x] 1.2 Implement validation for feature/metadata row alignment, exactly two groups, at least two stages, complete group-by-stage cells, fitted-model feature dimensions, and deterministic explicit-or-sorted level ordering.
- [x] 1.3 Implement default arithmetic group-by-stage means and an optional precomputed complete mean-table input for covariate-adjusted LS means, preserving canonical feature order and identifying the mean source.
- [x] 1.4 Export the new public attribution types and entry points from `src/motco/stats/__init__.py` without changing existing statistics APIs.

## 2. Directional PLS decomposition

- [x] 2.1 Implement adjacent-transition vectors, path lengths, unit directions, and signed two-group directional contrasts with a configurable zero-vector tolerance and explicit unavailable values.
- [x] 2.2 Implement shared-model score projection and inverse reconstruction for each group-stage mean, producing observed, PLS-captured, and observed-minus-captured transition and directional-contrast records.
- [x] 2.3 Preserve raw vectors alongside normalized contrasts and include fitted component count, feature order, group/stage order, and transition identifiers in machine-readable metadata.
- [x] 2.4 Implement optional original-unit conversion from validated per-feature scales, keeping standardized and original-unit effects separate and leaving PLS geometry in standardized coordinates.

## 3. Stability and biological aggregation

- [x] 3.1 Implement seeded within-group-by-stage bootstrap resampling that preserves source stratum sizes and reuses the frozen model and feature scales for every replicate.
- [x] 3.2 Compute bootstrap sign stability, rank stability, top-k selection frequency, valid-replicate counts, and effective bootstrap configuration for each transition and attribution component.
- [x] 3.3 Validate caller-supplied feature-to-module/pathway labels and aggregate signed effects, absolute effects, feature counts, and available stability summaries while recording the caller-supplied label source.
- [x] 3.4 Ensure zero bootstrap count returns explicit unavailable stability fields and that zero or near-zero effects do not create misleading sign stability denominators.

## 4. Tests

- [x] 4.1 Add focused unit tests for design validation, deterministic ordering, arithmetic and precomputed means, adjacent transitions, normalized signed contrasts, zero-vector handling, and model-dimension errors.
- [x] 4.2 Add reconstruction tests using a fitted PLS model that verify shared-model use, observed/captured/residual outputs, raw-vector retention, and no replacement of observed features by low-rank reconstructions.
- [x] 4.3 Add tests for standardized/original-unit effects, invalid scale validation, deterministic stratified bootstrap behavior, stability summaries, zero-bootstrap unavailability, and incomplete/conflicting feature mappings.
- [x] 4.4 Add tests for module/pathway aggregation and a multi-stage example with one row per adjacent transition, including a precomputed covariate-adjusted mean-table path.

## 5. Integration and documentation

- [x] 5.1 Add a findings-ready tabular/report helper or reproducible driver that writes feature-level, transition-level, aggregate, configuration, and interpretation-boundary outputs without fabricating significance values.
- [x] 5.2 Update the relevant API and trajectory documentation with the attribution workflow, standardized/original-unit distinction, bootstrap conditioning, module/pathway label contract, and non-causal interpretation boundary.
- [x] 5.3 Add a small deterministic end-to-end example or test showing how an upstream significant orientation result can be passed to attribution without refitting separate group latent spaces.

## 6. Verification

- [x] 6.1 Run the focused attribution, PLS, trajectory, preprocessing, and evaluation tests and resolve regressions.
- [x] 6.2 Run Ruff and mypy on the changed source and tests.
- [x] 6.3 Run strict OpenSpec validation and confirm every requirement scenario is covered by an implementation task and test or documented output.
