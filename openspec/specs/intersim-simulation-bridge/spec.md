## Purpose

Provide a Python-facing bridge that invokes the R InterSIM package and returns aligned semi-synthetic methylation, gene expression, protein expression, clustering, sample ID, and metadata outputs for downstream MOTCO simulation studies.

## Requirements

### Requirement: InterSIM availability can be checked from Python
MOTCO SHALL provide a Python API that checks whether `Rscript` is available and whether the R `InterSIM` package can be loaded.

#### Scenario: InterSIM is available
- **WHEN** the host has `Rscript` on `PATH` and `requireNamespace("InterSIM", quietly = TRUE)` succeeds
- **THEN** the availability check reports that InterSIM can be used

#### Scenario: Rscript is missing
- **WHEN** the host does not have `Rscript` on `PATH`
- **THEN** the availability check reports that InterSIM cannot be used with a message naming the missing `Rscript` dependency

#### Scenario: InterSIM package is missing
- **WHEN** `Rscript` is available but the R `InterSIM` package cannot be loaded
- **THEN** the availability check reports that InterSIM cannot be used with a message naming the missing R package

### Requirement: Python can invoke InterSIM
MOTCO SHALL provide a Python API that invokes R InterSIM with supported parameters and returns the generated data to Python.

#### Scenario: Successful simulation invocation
- **WHEN** a caller invokes the bridge with valid InterSIM parameters and InterSIM is available
- **THEN** the bridge returns methylation, gene expression, protein expression, sample IDs, cluster assignments, and metadata

#### Scenario: Invalid InterSIM parameters
- **WHEN** a caller invokes the bridge with parameters rejected by InterSIM
- **THEN** the bridge raises a Python exception that includes the R error output

### Requirement: Returned InterSIM matrices are aligned
The InterSIM bridge SHALL return omics matrices and cluster assignments with matching sample rows.

#### Scenario: Matrix row counts match cluster assignments
- **WHEN** a simulation completes successfully
- **THEN** methylation, gene expression, protein expression, and cluster assignments have the same number of rows

#### Scenario: Sample IDs are preserved
- **WHEN** a simulation completes successfully
- **THEN** all returned matrices and the cluster assignment use the same sample ID order

### Requirement: InterSIM simulation is reproducible by seed
The InterSIM bridge SHALL accept an explicit seed and use it to seed the R simulation process.

#### Scenario: Same seed produces identical outputs
- **WHEN** the same supported InterSIM parameters and seed are used twice in the same environment
- **THEN** the returned methylation, gene expression, protein expression, and cluster assignments are identical

#### Scenario: Seed is recorded in metadata
- **WHEN** a simulation completes successfully
- **THEN** the returned metadata includes the seed used for the R simulation

### Requirement: Bridge exposes InterSIM-native parameters
The InterSIM bridge SHALL support a Python parameter surface corresponding to InterSIM's native simulation arguments.

#### Scenario: Native parameters are translated
- **WHEN** a caller provides Python parameters for sample count, cluster proportions, omics effect sizes, differential feature proportions, covariance options, cross-omic correlations, and seed
- **THEN** the bridge translates those parameters to the corresponding R InterSIM invocation

#### Scenario: Omitted optional parameters use InterSIM defaults
- **WHEN** a caller omits optional InterSIM parameters
- **THEN** the bridge invokes InterSIM so that the R package defaults apply
