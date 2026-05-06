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
The harness SHALL support initial integration methods that can be implemented with existing MOTCO dependencies.

#### Scenario: Concatenated integration
- **WHEN** the caller selects `concat` integration
- **THEN** the harness creates an outcome matrix by combining aligned omics matrices with documented scaling behavior

#### Scenario: SNF integration
- **WHEN** the caller selects `snf` integration
- **THEN** the harness creates an outcome matrix from SNF fusion and spectral embedding

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
