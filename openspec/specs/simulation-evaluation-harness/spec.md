# simulation-evaluation-harness Specification

## Purpose
Define the per-replicate evaluation layer that runs one semi-synthetic trajectory dataset through MOTCO integration, trajectory statistics, optional RRPP p-value estimation, and result metadata capture.

## Requirements
### Requirement: Harness evaluates one semi-synthetic trajectory dataset
MOTCO SHALL provide a simulation evaluation harness that accepts one `SemiSyntheticTrajectoryDataset` and returns a structured evaluation result.

#### Scenario: Successful single-dataset evaluation
- **WHEN** a caller provides a valid semi-synthetic trajectory dataset and evaluation parameters
- **THEN** the harness returns observed trajectory statistics and evaluation metadata

#### Scenario: Result includes generator truth
- **WHEN** evaluation succeeds
- **THEN** the result includes the generator truth metadata from the input dataset

### Requirement: Harness supports initial integration methods
The harness SHALL construct the molecular latent space — the measurement substrate in which trajectory geometry is estimated — via a selectable integration method, operating on M-value-converted methylation and raw expression/proteomics. The production latent-space methods are **SNF** (graph-spectral embedding) and **PLS** (the transform of the omic features into the subspace that maximizes covariance with the stage label). `concat` is retained as a **baseline/diagnostic** path (standardized feature concatenation), not a constructed latent space. The viz down-projection (`plot_trajectory_from_*`) is display-only and distinct from this measurement space.

#### Scenario: Concatenated baseline integration
- **WHEN** the caller selects `concat` integration
- **THEN** the harness converts methylation to M-values, standardises all layers, and concatenates them into the outcome matrix
- **AND** the result metadata identifies `concat` as a baseline rather than a constructed latent space

#### Scenario: SNF integration
- **WHEN** the caller selects `snf` integration
- **THEN** the harness converts methylation to M-values and creates the latent space from SNF fusion and spectral embedding

#### Scenario: PLS integration
- **WHEN** the caller selects `pls` integration
- **THEN** the harness converts methylation to M-values, standardises all layers, fits PLS-DA conditioned on the stage label, and returns the PLS X-score matrix as the latent space
- **AND** the number of latent variables is selected by the double nested cross-validation (modal LV across repeats, parsimony tie-break) to secure a stable, non-overfitted molecular space
- **AND** the result metadata records the selected number of latent variables and the cross-validation parameters

#### Scenario: PLS integration is infeasible
- **WHEN** the caller selects `pls` integration but the sample provides too few observations per stage for the cross-validation
- **THEN** the harness raises a clear validation error

#### Scenario: Unsupported integration method
- **WHEN** the caller selects an unsupported integration method
- **THEN** the harness raises a clear validation error

### Requirement: Harness builds MOTCO trajectory design objects
The harness SHALL construct model matrices, LS means, and trajectory contrasts from generated sample metadata.

#### Scenario: Design objects are derived from metadata
- **WHEN** the dataset metadata contains valid `group` and `stage` columns
- **THEN** the harness builds full and reduced model matrices, LS means, and a two-group trajectory contrast

#### Scenario: Missing metadata columns are rejected
- **WHEN** required metadata columns are missing
- **THEN** the harness raises a clear validation error

### Requirement: Harness estimates observed trajectory differences
The harness SHALL estimate observed `deltas`, `angles`, and `shapes` using MOTCO trajectory routines.

#### Scenario: Observed statistics are returned
- **WHEN** evaluation succeeds
- **THEN** the result includes observed `deltas`, `angles`, and `shapes`

#### Scenario: Pairwise group statistic is exposed
- **WHEN** evaluation succeeds for two groups
- **THEN** the result includes scalar statistics for the generated group comparison

### Requirement: Harness optionally runs RRPP p-value estimation
The harness SHALL optionally run RRPP and compute empirical p-values for observed trajectory statistics.

#### Scenario: RRPP disabled
- **WHEN** the caller sets permutation count to 0
- **THEN** the harness returns observed statistics without RRPP p-values

#### Scenario: RRPP enabled
- **WHEN** the caller sets permutation count greater than 0
- **THEN** the harness runs RRPP and returns empirical p-values for available statistics

#### Scenario: P-values use plus-one correction
- **WHEN** RRPP p-values are computed
- **THEN** the harness uses `(1 + count(null >= observed)) / (1 + n_permutations)`

### Requirement: Harness records evaluation metadata
The harness SHALL record parameters and runtime metadata needed for later grid aggregation.

#### Scenario: Evaluation metadata is returned
- **WHEN** evaluation succeeds
- **THEN** the result includes integration method, integration parameters, permutation count, runtime seconds, and evaluation parameters
