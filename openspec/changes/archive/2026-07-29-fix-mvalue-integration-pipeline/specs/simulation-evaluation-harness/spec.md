## MODIFIED Requirements

### Requirement: Harness supports initial integration methods
The harness SHALL support integration methods that operate on M-value-converted methylation and raw expression/proteomics.

#### Scenario: Concatenated integration
- **WHEN** the caller selects `concat` integration
- **THEN** the harness converts methylation to M-values, standardises all layers, and concatenates them into the outcome matrix

#### Scenario: SNF integration
- **WHEN** the caller selects `snf` integration
- **THEN** the harness converts methylation to M-values and creates the outcome matrix from SNF fusion and spectral embedding

#### Scenario: PLS integration
- **WHEN** the caller selects `pls` integration
- **THEN** the harness converts methylation to M-values, standardises all layers, and builds the latent space from PLS-DA conditioned on the stage label

#### Scenario: Unsupported integration method
- **WHEN** the caller selects an unsupported integration method
- **THEN** the harness raises a clear validation error