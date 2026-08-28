## ADDED Requirements

### Requirement: Study provides a fixed Phase 4 medium-pilot profile

The study SHALL provide a version-controlled Phase 4 configuration using the numpy generator, pooled PLS integration with M-value methylation, 300 samples, four stages, 100 replicates per cell, 199 RRPP permutations, trajectory modes `magnitude`, `orientation`, `shape`, and `translation`, and effect sizes `0.00`, `0.25`, `0.50`, `0.75`, and `1.00`. The configuration MUST explicitly record PLS cross-validation parameters, seed policy, diagnostic settings, study intent, and an acceptance block holding every gate parameter: alpha, tolerance multipliers, minimum power at the top effect, confirmation-rule thresholds, and which mode/statistic pairs are mandatory versus descriptive.

#### Scenario: Phase 4 config is loaded

- **WHEN** the committed Phase 4 configuration is loaded
- **THEN** it deterministically enumerates the expected Type I controls and primary power cells with the declared sample size, stage count, replicate count, permutations, modes, and effects
- **AND** it requires no R runtime dependency

#### Scenario: Primary cells use matched seeds

- **WHEN** primary cells are enumerated for different modes or nonzero effects at the same replicate index
- **THEN** their generated datasets use the same matched replicate seed under a versioned seed-pairing policy
- **AND** the policy is included in parameter signatures so incompatible legacy shards cannot resume into the Phase 4 run

#### Scenario: Zero-effect cells are not duplicated across modes

- **WHEN** the Phase 4 configuration is enumerated
- **THEN** exactly one mode-agnostic zero-effect primary cell is emitted, inside the matched-seed family, and every trajectory mode's power curve resolves its `0.00` point from that shared anchor
- **AND** no two enumerated primary cells generate identical datasets at the same replicate index

#### Scenario: Gate parameters come from the configuration

- **WHEN** Phase 4 gate evaluation runs
- **THEN** every threshold it applies is read from the configuration's acceptance block
- **AND** no mandatory gate threshold is hard-coded in summary or report code

#### Scenario: Orientation diagnostics are selected without significance conditioning

- **WHEN** a primary orientation cell has a nonzero requested effect
- **THEN** its evaluation enables 100 frozen-model attribution bootstrap replicates with `top_k=20`
- **AND** eligibility does not depend on the observed global orientation p-value

### Requirement: Replicate persistence includes Phase 4 diagnostics

The study SHALL persist JSON-safe PLS integration metadata and optional attribution diagnostics beside existing p-values, observed statistics, realized geometry, truth metadata, and runtime metadata. Readers MUST remain able to load legacy records that lack the additive fields, while resume validation MUST reject records created with a different diagnostic configuration or schema version.

#### Scenario: Phase 4 replicate round-trips

- **WHEN** a completed Phase 4 replicate is serialized and loaded
- **THEN** its selected PLS component count, effective cross-validation metadata, realized-geometry diagnostics, and applicable attribution diagnostics are preserved

#### Scenario: Ineligible cell is persisted

- **WHEN** a cell is not selected for attribution diagnostics
- **THEN** its record explicitly identifies attribution as not requested rather than failed or unavailable

#### Scenario: Legacy record is loaded

- **WHEN** a reader loads a record written before Phase 4 diagnostic fields existed
- **THEN** the record loads with empty diagnostic fields
- **AND** it cannot satisfy resume for a Phase 4 cell with a different parameter signature

### Requirement: Phase 4 reporting relates operating characteristics to realized geometry

The study SHALL report per-statistic rejection rates and Monte Carlo uncertainty together with summaries of every applicable realized-geometry checkpoint. It MUST interpret off-diagonal rejection in light of the corresponding pre-integration geometry and MUST NOT label a response as estimator cross-talk solely because the requested mode name differs from the responding statistic.

#### Scenario: Geometry-aware operating report is produced

- **WHEN** merged Phase 4 records are reported
- **THEN** structured tables summarize path lengths, `delta`, `angle`, and `shape` by mode, effect, checkpoint, and scope beside the rejection-rate tables
- **AND** the report identifies the first checkpoint where each material off-diagonal response appears

#### Scenario: Measurement spaces are compared

- **WHEN** feature-space and PLS-latent checkpoints appear in one report
- **THEN** the report labels their measurement spaces and compares trends or retention without treating raw distances as scale-equivalent

#### Scenario: PLS and attribution stability are reported

- **WHEN** eligible PLS orientation replicates are available
- **THEN** the report summarizes selected component counts, attribution availability, observed-versus-captured retention, cross-replicate top-k selection and sign agreement, bootstrap stability, and generator-truth recovery by effect and transition

### Requirement: Phase 4 emits an explicit gate for the paper-grade study

The Phase 4 report SHALL evaluate predeclared mandatory gates and emit `proceed`, `hold`, or `indeterminate` for Phase 5. It MUST NOT emit `proceed` unless every mandatory gate is met with complete eligible diagnostics.

#### Scenario: Type I inflation gate is evaluated

- **WHEN** the `none` baseline and every translation cell at each enumerated effect level are complete
- **THEN** each available statistic is checked against the predeclared one-sided Monte Carlo inflation tolerance at alpha `0.05`

#### Scenario: A single control statistic is marginally above its bound

- **WHEN** exactly one control statistic exceeds its inflation bound by less than one Monte Carlo standard error and no other mandatory gate fails
- **THEN** the report emits `indeterminate`, names the cell and statistic, and states the confirmation re-run required before a Phase 5 decision
- **AND WHEN** two or more control statistics exceed their bounds, or any exceedance is at least one Monte Carlo standard error
- **THEN** the report emits `hold`

#### Scenario: Magnitude gate is evaluated

- **WHEN** magnitude cells are complete
- **THEN** `delta` power is checked for an uncertainty-tolerant non-decreasing curve and power of at least `0.80` at effect `1.00`
- **AND** magnitude `angle` and `shape` rejection rates are checked against the Type I inflation tolerance

#### Scenario: Orientation and shape gates are evaluated

- **WHEN** orientation and shape cells are complete
- **THEN** orientation `angle` and shape `shape` power are checked for uncertainty-tolerant non-decreasing curves and power of at least `0.80` at effect `1.00`
- **AND** off-diagonal responses are reported against realized geometry without imposing a false purity requirement on the mixed constructions
- **AND** the shape gate constrains only the diagonal `shape` response, reported against the validated invariance contract, so it tests detectability of the constructed bend rather than shape-specific purity

#### Scenario: Diagnostic completeness gate is evaluated

- **WHEN** all expected work units have been merged
- **THEN** every completed PLS replicate is required to contain selected-component and realized-geometry metadata
- **AND** every eligible orientation replicate is required to contain valid attribution diagnostics or a recorded failure reason

#### Scenario: A mandatory gate fails or lacks evidence

- **WHEN** any mandatory gate fails
- **THEN** the report emits `hold` with the failed criteria and supporting observations
- **AND WHEN** expected records or required diagnostics are incomplete
- **THEN** the report emits `indeterminate` and does not recommend the paper-grade study

### Requirement: Phase 4 findings are versioned and reproducible

The study SHALL produce a dated findings report tied to the exact configuration, code revision, parameter signatures, software versions, record counts, failure counts, and reproduction commands. The prior July pilot and its outputs MUST remain unchanged and be identified as superseded for Phase 4 gate decisions.

#### Scenario: Findings report is committed

- **WHEN** the Phase 4 run and reporting complete
- **THEN** a versioned report records the gate decision, scientific interpretation, limitations, and exact shard, merge, and report commands
- **AND** all claims in the report can be traced to structured CSV or JSON outputs

