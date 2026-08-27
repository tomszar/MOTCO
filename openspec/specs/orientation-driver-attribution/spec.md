# orientation-driver-attribution Specification

## Purpose

This capability explains a detected group trajectory orientation difference in a shared PLS molecular space by identifying signed feature contrasts, the portion retained by PLS, omitted residual structure, and stability of the resulting drivers.

## Requirements

### Requirement: Attribution uses one shared PLS coordinate system

The system SHALL accept one aligned feature matrix, group and stage metadata, and one fitted pooled PLS model, and SHALL use that same model for both groups and every transition. It MUST NOT fit separate group-specific PLS models during attribution.

#### Scenario: Valid shared-model attribution

- **WHEN** the feature rows, metadata rows, feature order, and fitted model input dimensions agree and metadata contains exactly two groups and at least two stages
- **THEN** the system returns an attribution result using the shared model

#### Scenario: Invalid alignment is rejected

- **WHEN** feature rows, metadata rows, sample identifiers, or model input dimensions do not agree
- **THEN** the system raises a descriptive validation error before calculating driver values

#### Scenario: Unsupported design is rejected

- **WHEN** metadata contains fewer than two groups, more than two groups, fewer than two stages, or a missing group-by-stage cell
- **THEN** the system raises a descriptive validation error naming the invalid design

### Requirement: Attribution separates direction from transition magnitude

The system SHALL calculate, for each group and every adjacent ordered stage transition, the feature transition vector, its Euclidean norm, its unit direction when the norm is nonzero, and the signed directional contrast between the two groups. The group contrast SHALL use a documented deterministic group ordering and SHALL preserve feature signs.

#### Scenario: Two-stage directional contrast

- **WHEN** two groups have nonzero transitions between two stages
- **THEN** the result contains one directional contrast equal to the second ordered group's unit transition minus the first ordered group's unit transition, together with both group path lengths

#### Scenario: Multi-stage adjacent transitions

- **WHEN** the design contains three or more ordered stages
- **THEN** the result contains a separate directional contrast and path length pair for every adjacent stage transition

#### Scenario: Zero transition is unavailable

- **WHEN** either group's transition norm is zero or below the configured numerical tolerance
- **THEN** direction-dependent values for that transition are marked unavailable and the result retains the raw transition and path length information

### Requirement: Attribution decomposes observed direction into PLS-captured and residual components

The system SHALL compute observed group-stage feature means, project those means through the shared PLS model, reconstruct the retained feature-space component, and report observed, PLS-captured, and observed-minus-captured directional contrasts for every applicable transition. The decomposition SHALL use a fixed model and SHALL not replace observed features with low-rank reconstructions.

#### Scenario: PLS-captured contrast is reported

- **WHEN** a valid nonzero transition is available for both groups
- **THEN** the result contains observed and reconstructed PLS directional contrasts, their feature-level signed values, and a residual contrast defined as observed minus reconstructed

#### Scenario: Model remains frozen

- **WHEN** attribution calculates observed values, PLS reconstructions, or bootstrap replicates
- **THEN** the fitted PLS model and its coordinate system remain unchanged

#### Scenario: Reconstruction is unavailable for an invalid model

- **WHEN** the fitted model cannot transform or inverse-transform the aligned feature matrix
- **THEN** the system raises a descriptive attribution error rather than returning partial driver values

### Requirement: Results distinguish standardized and original-unit effects

The system SHALL report feature-level attribution in the input coordinate units and SHALL optionally report a second set of observed, PLS-captured, and residual effects in original feature units when valid per-feature scales are supplied. Unit labels and feature order SHALL be retained in the result.

#### Scenario: Standardized effects are returned

- **WHEN** attribution is run on the pooled standardized feature matrix
- **THEN** every feature-level effect identifies that it is in standardized input units

#### Scenario: Original-unit effects are requested

- **WHEN** positive finite per-feature scales matching the input feature order are supplied
- **THEN** the result reports corresponding original-unit effects without changing the standardized PLS geometry

#### Scenario: Invalid unit conversion is rejected

- **WHEN** supplied feature scales are missing, non-finite, non-positive, duplicated, or differently ordered from the input features
- **THEN** the system raises a descriptive validation error

### Requirement: Bootstrap estimates driver stability within design strata

The system SHALL optionally resample subjects with replacement independently within each group-by-stage stratum, while keeping the fitted preprocessing contract and PLS model fixed, and SHALL summarize feature sign stability, rank stability, and top-k selection frequency for each transition and attribution component.

#### Scenario: Stratified bootstrap is reproducible

- **WHEN** attribution is run twice with the same data, bootstrap count, and seed
- **THEN** the bootstrap summaries are identical

#### Scenario: Bootstrap preserves design strata

- **WHEN** a bootstrap replicate is generated
- **THEN** every group-by-stage stratum contributes the same number of resampled observations as the source stratum and no observation is sampled across strata

#### Scenario: Stability summaries are returned

- **WHEN** bootstrap resampling is enabled with a positive replicate count
- **THEN** the result includes per-feature sign stability, rank stability, top-k selection frequency, bootstrap configuration, and the number of valid replicates

#### Scenario: Bootstrap is disabled explicitly

- **WHEN** the bootstrap count is zero
- **THEN** no resampling is performed and stability fields are marked unavailable rather than treated as zero stability

### Requirement: Caller-supplied feature groups can be aggregated

The system SHALL optionally accept validated caller-supplied feature-to-group labels for correlated-feature modules or biological pathways and SHALL return group-level summaries of signed and absolute attribution, feature counts, and bootstrap stability when available. The system MUST NOT imply that the labels came from an internal pathway database or that aggregation establishes causality.

#### Scenario: Module aggregation

- **WHEN** multiple features share a supplied module label
- **THEN** the result contains one summary row per module and transition for observed, PLS-captured, and residual attribution

#### Scenario: Pathway labels are preserved

- **WHEN** supplied labels identify pathways
- **THEN** the result retains the caller-provided pathway names and records the label source as caller-supplied

#### Scenario: Incomplete mapping is rejected

- **WHEN** a supplied feature-group mapping omits an input feature, contains an unknown feature, or assigns conflicting labels
- **THEN** the system raises a descriptive validation error

### Requirement: Attribution output is machine-readable and interpretation-bounded

The system SHALL return structured results that retain group and stage ordering, transition identifiers, feature names, model/component metadata, effective bootstrap parameters, and unit metadata. The output SHALL state that attribution describes associations implied by the fitted shared PLS representation and SHALL not claim a unique inverse, biological causality, or inferential significance.

#### Scenario: Result can be consumed without parsing prose

- **WHEN** a valid attribution analysis completes
- **THEN** feature-level, transition-level, aggregate, and configuration data are available as structured records or tables

#### Scenario: Significance is not fabricated

- **WHEN** attribution is run without an external hypothesis-test result
- **THEN** the result contains no synthetic p-value or significance decision and leaves selection of significant global results to the calling workflow
