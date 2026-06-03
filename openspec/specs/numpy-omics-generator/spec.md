# numpy-omics-generator Specification

## Purpose
TBD - created by archiving change numpy-feature-surgery-generator. Update Purpose after archive.
## Requirements
### Requirement: numpy generator reproduces InterSIM's mean-shift model
MOTCO SHALL provide a numpy-native multi-omic generator that produces methylation, gene expression, and proteomics matrices using InterSIM's generative model: each stage mean is `μ = base + δ · v` for a per-stage differential indicator `v ∈ {0,1}ᵖ`, with samples drawn from a multivariate normal with the reference covariance, and methylation passed through the inverse-logit (`rev.logit`) transform after the additive shift.

#### Scenario: Generates the three aligned omic matrices
- **WHEN** the generator is invoked with valid parameters
- **THEN** it returns methylation, expression, and proteomics matrices with the same number of rows, aligned to a shared sample order

#### Scenario: Methylation shift is applied in logit space
- **WHEN** a differential shift is applied to methylation features
- **THEN** the shift is added to the pre-logit (M-value) mean and the inverse-logit transform is applied last, so methylation values remain in `(0, 1)`

#### Scenario: Cross-omic coupling is preserved
- **WHEN** differential expression and proteomics features are derived by default from the differential methylation features
- **THEN** the generator maps differential CpGs to genes and genes to proteins using the cached cross-omic maps

### Requirement: Generation uses cached InterSIM reference data with provenance
The generator SHALL source its reference means, covariances, and cross-omic maps from a cached artifact committed to the repository, recorded with provenance, and reproducible by a one-time export.

#### Scenario: Reference cache is loaded without R
- **WHEN** the generator runs
- **THEN** it loads the reference means, covariances, and maps from the cached artifact without invoking R

#### Scenario: Provenance is recorded
- **WHEN** the reference cache is produced
- **THEN** it records the InterSIM version, export date, and the export procedure used to create it

### Requirement: Generation is reproducible by seed and requires no R at runtime
The generator SHALL be deterministic given its seed and parameters, and SHALL NOT require an R runtime.

#### Scenario: Same seed produces identical matrices
- **WHEN** the generator is invoked twice with the same seed and parameters
- **THEN** the returned matrices are identical

#### Scenario: Runs without Rscript on PATH
- **WHEN** the generator is invoked in an environment with no `Rscript`
- **THEN** generation succeeds using only the cached reference data

### Requirement: Generator realism is validated against InterSIM
The generator's output distributions SHALL be validated to match InterSIM's within tolerance, as a guard against reimplementation drift.

#### Scenario: Output distributions match InterSIM within tolerance
- **WHEN** the numpy generator and InterSIM are run with matched parameters and seed handling
- **THEN** per-omic feature means, covariance structure, cluster separation, and cross-omic correlation agree within a documented tolerance

### Requirement: Generator exposes per-stage differential indicators
The generator SHALL expose, as output, the differential-feature indicator used for each stage (and group) so downstream consumers can use the ground truth.

#### Scenario: Differential indicators are returned per stage
- **WHEN** generation succeeds
- **THEN** the per-stage differential-feature indicators for each omic layer are available in the returned truth structure

