# trajectory-power-study Specification

## Purpose
Define a declaratively-configured, cluster-executable study that characterizes the Type I error and power of the MOTCO trajectory test via per-statistic operating characteristics, with negative controls, pre-specified acceptance targets, and paper-ready reporting (specificity matrix, Type I tables, power curves).

## Requirements

### Requirement: Study is defined by a declarative configuration

MOTCO SHALL provide a declarative, file-based study configuration that fully determines an enumerated grid of simulation cells, so that a study is reproducible from its config alone.

The configuration MUST capture baseline InterSIM/generator/evaluation parameters, the set of trajectory modes, the set of effect sizes, the one-factor-at-a-time axes with their values, the per-cell replicate count, the base seed, and the pre-specified acceptance targets.

#### Scenario: Config enumerates a deterministic grid

- **WHEN** a study configuration is loaded and enumerated
- **THEN** it produces a `SimulationGrid` of Type I and power cells with stable, deterministic cell identifiers
- **AND** enumerating the same configuration twice yields identical cell identifiers and parameter signatures

#### Scenario: Config records negative-control modes

- **WHEN** a study configuration is enumerated
- **THEN** the resulting Type I cells include both the `none` (no group effect) and `translation` (location-only group effect) trajectory modes as negative controls

#### Scenario: Invalid configuration is rejected

- **WHEN** a study configuration omits a required field or specifies an unknown trajectory mode, a negative replicate count, or an unknown axis namespace
- **THEN** loading the configuration raises a clear validation error identifying the offending field

### Requirement: Study executes as resumable per-shard work units

MOTCO SHALL execute an enumerated study as `(cell, replicate)` work units partitioned across a fixed number of shards, so that the study can run as parallel cluster tasks without coordination.

Each shard MUST persist its own JSONL output file and MUST be independently resumable using the existing parameter-signature guard, so re-running a shard skips already-completed replicates and never appends duplicates.

#### Scenario: Work is partitioned deterministically across shards

- **WHEN** a study is run with `n_shards = N` and a given shard index `i`
- **THEN** the shard executes exactly the `(cell, replicate)` units assigned to index `i` by a deterministic partition of all units
- **AND** the union of units across all `N` shards equals the full set of units with no overlaps

#### Scenario: Shard resumes without duplicating completed work

- **WHEN** a shard is re-run and its JSONL output already contains completed records with matching parameter signatures
- **THEN** the shard skips those replicates and appends only missing or failed ones

#### Scenario: Shard records failures without aborting the study

- **WHEN** a replicate within a shard fails and the configured error policy is to record
- **THEN** the shard writes a failed replicate record with error details and continues with remaining units

### Requirement: Shards merge into a single deduplicated result set

MOTCO SHALL merge per-shard JSONL outputs into a single result set, deduplicating by `(cell_id, replicate_index)` and validating parameter-signature consistency across shards.

#### Scenario: Merge combines all shards

- **WHEN** per-shard JSONL files are merged
- **THEN** the merged result contains exactly one record per `(cell_id, replicate_index)` across all shards

#### Scenario: Merge detects inconsistent shards

- **WHEN** two shards contain the same `(cell_id, replicate_index)` with different parameter signatures
- **THEN** the merge raises a clear validation error rather than silently choosing one record

### Requirement: Study characterizes each statistic independently

MOTCO SHALL characterize the trajectory test using per-statistic operating characteristics, reporting the rejection rate of each of the `delta`, `angle`, and `shape` statistics as its own marginal quantity, without multiplicity correction across statistics.

#### Scenario: Per-statistic rejection rates are reported per cell

- **WHEN** merged results are summarized for a cell
- **THEN** the summary reports, for each of `delta`, `angle`, and `shape`, the completed replicate count, rejection count, rejection rate, and Monte Carlo standard error at the configured alpha

#### Scenario: Unavailable statistic is not counted as a rejection

- **WHEN** a statistic is unavailable for a cell (for example `shape` with fewer than three stages)
- **THEN** the summary reports it as unavailable rather than treating it as non-significant

### Requirement: Study reports a combined-rule Type I result

MOTCO SHALL report, as a secondary result, the Type I error rate of the combined decision rule that rejects when any of the three statistics is significant at the configured alpha, computed only over null cells.

#### Scenario: Combined-rule false-positive rate is computed on null cells

- **WHEN** a null cell (negative control) is summarized under the combined rule
- **THEN** a replicate counts as a rejection if any available statistic's p-value is below alpha
- **AND** the reported rate is the fraction of such replicates with its Monte Carlo standard error

### Requirement: Study produces paper-ready reports

MOTCO SHALL produce, from the merged and summarized results, a mode × statistic specificity matrix, Type I tables, and power-curve data, written as CSV and as figures.

#### Scenario: Specificity matrix is produced

- **WHEN** reporting runs on summarized results
- **THEN** it produces a matrix indexed by trajectory mode and statistic whose entries are rejection rates with Monte Carlo standard errors

#### Scenario: Power curves are produced

- **WHEN** reporting runs on power cells
- **THEN** it produces, for each trajectory mode and statistic, rejection rate as a function of effect size, suitable for plotting as a curve with error bars

#### Scenario: Type I table is produced

- **WHEN** reporting runs on null cells across the configured axes
- **THEN** it produces a table of per-statistic and combined-rule rejection rates with Monte Carlo standard errors

### Requirement: Study evaluates results against pre-specified targets

MOTCO SHALL evaluate summarized results against the acceptance targets declared in the configuration and report, per target, whether it is met given Monte Carlo uncertainty.

#### Scenario: Type I control target is evaluated

- **WHEN** a null cell is evaluated against a Type I control target at alpha
- **THEN** the report indicates whether the empirical rejection rate is within the target's tolerance (expressed in Monte Carlo standard errors) of alpha

#### Scenario: Power monotonicity target is evaluated

- **WHEN** a power mode's diagonal statistic is evaluated against a monotonicity target
- **THEN** the report indicates whether the rejection rate is non-decreasing in effect size and reaches the target's minimum power at the largest effect size

#### Scenario: Specificity target is evaluated

- **WHEN** an off-diagonal mode × statistic combination is evaluated against a specificity target
- **THEN** the report indicates whether its rejection rate stays within tolerance of alpha

### Requirement: Study runs on the numpy generator without an R runtime dependency
The trajectory power study SHALL generate every replicate through the numpy generator and cached reference data, so that study execution (including cluster shards) requires no `Rscript` or R `InterSIM` package.

#### Scenario: Shards run without R
- **WHEN** a study shard executes its `(cell, replicate)` units
- **THEN** each replicate is generated from the numpy generator and cached reference data, with no R invocation

#### Scenario: Negative-control modes are retained under the new semantics
- **WHEN** a study configuration is enumerated
- **THEN** the Type I cells still include `none` (no group effect) and `translation` (location-only group effect) as negative controls, now defined by the feature-surgery generator

### Requirement: Acceptance targets are re-specified for the new mode semantics
Because the trajectory modes are redefined, the study's pre-specified acceptance targets SHALL be reset to reflect the operating characteristics of the feature-surgery modes, and prior results SHALL be treated as superseded.

#### Scenario: Acceptance targets reference the new modes
- **WHEN** a study configuration's acceptance targets are evaluated
- **THEN** the per-statistic power and specificity targets correspond to the feature-surgery `magnitude`/`orientation`/`shape` modes

#### Scenario: Specificity demonstration is supported by indicator truth
- **WHEN** a replicate is summarized
- **THEN** the per-stage/group differential indicators emitted by the generator are available to confirm that the injected mode predominantly moves its matching statistic

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

### Requirement: Reporting stratifies orientation power by configuration eigengap

The study report SHALL summarize the recorded pooled and per-group eigengaps per cell beside the existing geometry summaries, and for orientation-mode power cells SHALL report rejection rates stratified by the recorded pooled eigengap, with per-stratum replicate counts and Monte Carlo uncertainty. Stratification MUST read recorded values only — the report MUST NOT regenerate datasets or recompute spectra from raw data.

#### Scenario: Eigengap summaries join the cell tables

- **WHEN** merged records carrying the spectrum block are reported
- **THEN** per-cell tables include summaries of the recorded pooled and per-group eigengaps

#### Scenario: Orientation power is stratified

- **WHEN** an orientation-mode power cell with recorded spectra is reported
- **THEN** the report includes rejection rates by eigengap stratum with per-stratum counts and uncertainty
- **AND** the strata are defined from the recorded values, not from regenerated data

#### Scenario: Legacy records degrade gracefully

- **WHEN** a report runs over merged records lacking the spectrum block
- **THEN** the stratified table is reported as unavailable for those cells and every pre-existing table is produced unchanged
