## MODIFIED Requirements

### Requirement: Harness supports initial integration methods
The harness SHALL construct the molecular latent space — the measurement substrate in which trajectory geometry is estimated — via a selectable integration method. The production latent-space methods are **SNF** (graph-spectral embedding) and **PLS** (the transform of the omic features into the subspace that maximizes covariance with the stage label). `concat` is retained as a **baseline/diagnostic** path (standardized feature concatenation), not a constructed latent space. The viz down-projection (`plot_trajectory_from_*`) is display-only and distinct from this measurement space.

#### Scenario: Concatenated baseline integration
- **WHEN** the caller selects `concat` integration
- **THEN** the harness creates an outcome matrix by combining aligned omics matrices with documented scaling behavior
- **AND** the result metadata identifies `concat` as a baseline rather than a constructed latent space

#### Scenario: SNF integration
- **WHEN** the caller selects `snf` integration
- **THEN** the harness creates the latent space from SNF fusion and spectral embedding

#### Scenario: PLS integration
- **WHEN** the caller selects `pls` integration
- **THEN** the harness standardizes and concatenates the omic blocks, fits PLS-DA conditioned on the stage label, and returns the PLS X-score matrix as the latent space
- **AND** the number of latent variables is selected by the double nested cross-validation (modal LV across repeats, parsimony tie-break) to secure a stable, non-overfitted molecular space
- **AND** the result metadata records the selected number of latent variables and the cross-validation parameters

#### Scenario: PLS integration is infeasible
- **WHEN** the caller selects `pls` integration but the sample provides too few observations per stage for the cross-validation
- **THEN** the harness raises a clear validation error

#### Scenario: Unsupported integration method
- **WHEN** the caller selects an unsupported integration method
- **THEN** the harness raises a clear validation error
