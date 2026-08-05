# latent-space-metric-compatibility Specification

## Purpose

Document the relationship between MOTCO's Euclidean trajectory statistics and
the geometry produced by each supported integration method.

## ADDED Requirements

### Requirement: Measurement-space assumptions are explicit

The documentation SHALL state that `delta`, `angle`, and Procrustes `shape`
measure Euclidean geometry in the supplied outcome matrix and SHALL distinguish
standardized concatenation, linear PCA/PLS projections, and SNF spectral
coordinates.

#### Scenario: User selects a linear projection

- **WHEN** a user interprets trajectory statistics calculated in PCA or PLS scores
- **THEN** the documentation explains that the scores are a lossy linear projection whose geometry can be related to the original features through fitted loadings

#### Scenario: User selects SNF

- **WHEN** a user interprets trajectory statistics calculated in SNF spectral coordinates
- **THEN** the documentation warns that feature-space path length, direction, and shape are not guaranteed to be preserved and recommends simulation validation

### Requirement: Graph-native alternatives are scoped accurately

The documentation SHALL identify diffusion, connectivity, neighbourhood, and
transition-profile quantities as geometrically aligned candidates for SNF and
SHALL state that MOTCO does not currently implement them.

#### Scenario: User seeks an SNF-native trajectory interpretation

- **WHEN** the user consults the SNF documentation
- **THEN** candidate graph-native metric families are listed without implying that they are available in the current API

### Requirement: Pilot claims distinguish integration from metric compatibility

The latent-space cross-talk findings SHALL describe the observed SNF result as
a property of SNF spectral coordinates combined with MOTCO's Euclidean
statistics, rather than a general failure of SNF as a multi-omics integration
method.

#### Scenario: Reader reviews the SNF pilot finding

- **WHEN** the reader encounters the reported lack of `delta`/`angle` sensitivity
- **THEN** the findings identify the metric mismatch and note that graph-native alternatives were not evaluated
