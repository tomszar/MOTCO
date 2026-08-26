## 1. Analytic Generator Truth

- [x] 1.1 Introduce a structured group-by-stage population-mean representation with canonical group, stage, omic, and feature ordering.
- [x] 1.2 Centralize the generator mean construction so sampling and analytic truth use the same methylation M-value, expression, and proteomics mean definitions.
- [x] 1.3 Attach analytic population trajectories to in-memory semi-synthetic truth while keeping high-dimensional arrays out of serialized replicate metadata.
- [x] 1.4 Add tests that population means reflect group-specific deltas, transformed indicators, and CpG-to-gene-to-protein cascade propagation, and agree with large-sample averages within Monte Carlo tolerance.

## 2. Shared Pre-Integration Preprocessing

- [x] 2.1 Add a fitted block-preprocessing representation that records canonical feature order, pooled feature means, effective scales, and the methylation M-value conversion contract.
- [x] 2.2 Implement aligned transforms for observed omic blocks and analytic population means, including deterministic handling of near-zero feature scales.
- [x] 2.3 Refactor concatenation and PLS integration to consume the shared transformed observed matrix without changing their existing numerical outputs or metadata semantics.
- [x] 2.4 Add regression tests proving diagnostic joint features equal the PLS input and existing concat/PLS scores and trajectory statistics remain stable.

## 3. Geometry Decomposition

- [x] 3.1 Define JSON-safe diagnostic structures for checkpoint, scope, group path lengths, pairwise `delta`, `angle`, `shape`, availability, and schema version.
- [x] 3.2 Implement geometry calculation from ordered group-stage population means and from observed data using the production design and trajectory estimands.
- [x] 3.3 Produce per-omic `population_native`, per-omic and joint `population_standardized`, per-omic and joint `observed_standardized`, and joint `pls_latent` diagnostics only where applicable.
- [x] 3.4 Handle fewer-than-three-stage shape and zero-path orientation as explicit unavailable values while preserving all other defined statistics.
- [x] 3.5 Add deterministic tests for known magnitude, orientation, bend, translation, zero-path, and two-stage configurations at each applicable checkpoint.

## 4. Evaluation and Study Persistence

- [x] 4.1 Extend the semi-synthetic evaluation result with nested realized-geometry diagnostics while preserving all existing result fields and meanings.
- [x] 4.2 Extend grid replicate serialization and loading to round-trip diagnostics and construction metadata without serializing population matrices or raw indicators.
- [x] 4.3 Version the diagnostic schema or parameter signature so resume validation rejects incompatible legacy shards while legacy records remain readable as lacking decomposition data.
- [x] 4.4 Add evaluation, JSONL round-trip, legacy-record, resume-mismatch, and unavailable-value aggregation tests.

## 5. Phase 2 Characterization

- [x] 5.1 Add a reproducible matched-seed Phase 2 driver covering orientation and both shape constructions over the roadmap effect-size grid, with magnitude, none, and translation controls.
- [x] 5.2 Add structured summaries that flatten checkpoint diagnostics for analysis and report path lengths, `delta`, `angle`, and `shape` without equating raw feature and latent scales.
- [x] 5.3 Run a small deterministic characterization to verify requested-effect monotonicity and locate the first checkpoint and omic scope where off-target responses appear.
- [x] 5.4 Write a committed Phase 2 findings report documenting the construction, cascade, preprocessing, sampling, and PLS contributions and recommending revised orientation and shape contracts.
- [x] 5.5 Update the scientific roadmap and relevant simulation documentation with the Phase 2 gate decision and exact reproduction commands.

## 6. Verification

- [x] 6.1 Run the focused simulation and trajectory test modules with reduced permutation settings where applicable.
- [x] 6.2 Run the fast test suite, Ruff, and mypy and resolve all regressions introduced by the change.
- [x] 6.3 Validate the OpenSpec change strictly and confirm all implemented requirements are covered by tests or the committed characterization report.
