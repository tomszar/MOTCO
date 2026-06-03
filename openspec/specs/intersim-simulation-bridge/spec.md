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

### Requirement: Bridge rejects out-of-range seeds before invoking R

The InterSIM bridge SHALL validate that the supplied seed is within
R's signed-32-bit range `[-2³¹, 2³¹ − 1]` and raise a clear Python
error before launching the R subprocess when it is not.

#### Scenario: Out-of-range positive seed is rejected with a clear Python error

- **WHEN** a caller invokes the bridge with a seed greater than
  `2³¹ − 1`
- **THEN** the bridge raises an `InterSIMError` (or subclass) whose
  message names the offending value and the accepted range, without
  starting the R subprocess

#### Scenario: Out-of-range negative seed is rejected with a clear Python error

- **WHEN** a caller invokes the bridge with a seed less than `-2³¹`
- **THEN** the bridge raises an `InterSIMError` (or subclass) whose
  message names the offending value and the accepted range, without
  starting the R subprocess

#### Scenario: In-range seed proceeds normally

- **WHEN** a caller invokes the bridge with a seed in
  `[-2³¹, 2³¹ − 1]`
- **THEN** the bridge launches the R subprocess and the seed reaches
  `set.seed` unchanged

### Requirement: Bridge can export InterSIM reference data for the numpy generator
The InterSIM bridge SHALL provide a one-time export path that captures InterSIM's reference means, covariances, and cross-omic maps into a cached artifact consumed by the numpy generator, so that R is needed only to produce the cache and not for runtime generation.

#### Scenario: Reference export captures the required objects
- **WHEN** the export path is run with InterSIM available
- **THEN** it writes the reference means, covariances, cross-omic maps, and correlation constants needed to reproduce InterSIM's generative model

#### Scenario: Export records provenance
- **WHEN** the export completes
- **THEN** the cached artifact records the InterSIM version and export date

#### Scenario: Runtime generation does not invoke the bridge
- **WHEN** datasets are generated for evaluation, the grid, the study, or the showcase
- **THEN** generation uses the numpy generator and the cached reference data, without invoking the InterSIM bridge or R

