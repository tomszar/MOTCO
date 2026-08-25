## Purpose

Define reproducible diagnostics that locate where requested semi-synthetic trajectory geometry changes across generator construction, sampling, pooled preprocessing, and PLS projection.

## ADDED Requirements

### Requirement: Generator exposes analytic population trajectories
The system SHALL expose the exact group-by-stage population mean trajectory for each omic layer in the units supplied to integration: M-values for methylation and native generated units for expression and proteomics.

#### Scenario: Population means are derived without sampling noise
- **WHEN** a semi-synthetic trajectory dataset is generated
- **THEN** its diagnostic truth contains population means derived from the generator baselines, group-specific differential indicators, and effect sizes
- **AND** those means do not depend on the sampled observations

#### Scenario: Biological cascade is represented
- **WHEN** a methylation construction changes the derived expression or protein indicators
- **THEN** the population trajectories for expression and proteomics reflect the corresponding CpG-to-gene-to-protein propagation

### Requirement: Pre-integration preprocessing is shared and reproducible
The system SHALL apply one pooled per-feature preprocessing contract to both observed omic matrices and analytic population means, converting methylation to M-values and standardizing each omic block before concatenation.

#### Scenario: Population and observations use identical fitted scaling
- **WHEN** standardized population and observed diagnostics are calculated for one replicate
- **THEN** both are transformed using the same feature order and the same pooled location and scale parameters fitted from that replicate's observed samples

#### Scenario: Constant features remain finite
- **WHEN** a fitted feature scale is zero or below the established numerical tolerance
- **THEN** the effective scale is replaced consistently for population and observed transformations
- **AND** all diagnostic matrices remain finite

#### Scenario: PLS receives the diagnostic joint matrix
- **WHEN** PLS integration is evaluated
- **THEN** the observed standardized joint matrix used by the diagnostics is identical to the feature matrix presented to the PLS fit

### Requirement: Diagnostics decompose realized trajectory geometry
The system SHALL report group-specific path lengths and pairwise `delta`, `angle`, and `shape` at every applicable geometry checkpoint and scope.

#### Scenario: Native population geometry is reported per omic
- **WHEN** diagnostics are calculated for a valid dataset
- **THEN** native population geometry is reported separately for methylation, expression, and proteomics
- **AND** no joint native-space statistic is reported across incompatible omic units

#### Scenario: Standardized population geometry is reported
- **WHEN** diagnostics are calculated for a valid dataset
- **THEN** standardized population geometry is reported for each omic and for their joint concatenation

#### Scenario: Observed standardized geometry is reported
- **WHEN** diagnostics are calculated for a valid dataset
- **THEN** observed geometry is estimated using the production group-stage design for each standardized omic and their joint concatenation

#### Scenario: PLS latent geometry is reported
- **WHEN** the evaluation method is PLS
- **THEN** geometry is reported for the observed PLS score trajectory using the same fitted PLS representation used by the trajectory test

#### Scenario: Shape is unavailable for fewer than three stages
- **WHEN** a trajectory contains fewer than three stages
- **THEN** the diagnostic marks `shape` as unavailable rather than reporting a numeric value

#### Scenario: Orientation is undefined for a zero-length trajectory
- **WHEN** either group has zero path length at a checkpoint
- **THEN** the diagnostic marks `angle` as unavailable and preserves the remaining defined statistics

### Requirement: Geometry diagnostics are structured and attributable
The system SHALL associate each diagnostic value with its checkpoint, omic scope, statistic, requested mode and effect, and generator construction metadata.

#### Scenario: Evaluation returns diagnostics
- **WHEN** a semi-synthetic trajectory evaluation succeeds
- **THEN** its structured result includes the realized-geometry diagnostics and the construction metadata needed to interpret them

#### Scenario: Replicate records preserve diagnostics
- **WHEN** a study replicate is serialized and subsequently loaded
- **THEN** all applicable checkpoints, scopes, path lengths, trajectory statistics, and unavailable-value markers are preserved

#### Scenario: Existing result fields remain available
- **WHEN** realized-geometry diagnostics are enabled
- **THEN** the existing observed statistics, p-values, truth metadata, and runtime metadata remain available with their existing meanings

### Requirement: Study summaries characterize construction behavior
The system SHALL summarize requested versus realized orientation and shape geometry across effect sizes and replicates without assuming that each requested mode affects only its namesake statistic.

#### Scenario: Orientation behavior is summarized
- **WHEN** orientation replicates span multiple requested effect sizes
- **THEN** the summary shows the realized path lengths, `delta`, `angle`, and `shape` by checkpoint and scope so monotonicity and the first source of cross-talk can be assessed

#### Scenario: Shape behavior is summarized
- **WHEN** shape replicates span multiple requested effect sizes
- **THEN** the summary distinguishes the requested bend from realized magnitude, orientation, and shape responses at each checkpoint

#### Scenario: Checkpoints are compared without equating their raw scales
- **WHEN** feature-space and PLS-latent results are summarized together
- **THEN** the report identifies their different measurement spaces and does not interpret raw distances as directly scale-equivalent
