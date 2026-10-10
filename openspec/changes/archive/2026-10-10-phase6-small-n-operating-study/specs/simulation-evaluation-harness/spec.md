# Spec Delta

## MODIFIED Requirements

### Requirement: Harness supports initial integration methods
The harness SHALL construct the molecular latent space — the measurement substrate in which trajectory geometry is estimated — via a selectable integration method, operating on M-value-converted methylation and raw expression/proteomics. The production latent-space methods are **SNF** (graph-spectral embedding) and **PLS** (the transform of the omic features into the subspace that maximizes covariance with the stage label). `concat` is retained as a **baseline/diagnostic** path (standardized feature concatenation), not a constructed latent space. The viz down-projection (`plot_trajectory_from_*`) is display-only and distinct from this measurement space.

An optional integration parameter SHALL select which omic blocks are measured: a non-empty, duplicate-free subset of methylation, expression, and proteomics. Only the selected blocks SHALL enter pooled preprocessing, the integration method, orientation attribution, and every realized-geometry checkpoint (population, standardized population, observed standardized, and latent). Every joint scope SHALL be computed over the selected blocks only, and per-block scopes SHALL be reported for the selected blocks only. Blocks that are not selected SHALL be generated but not measured. When the parameter is absent, all three blocks SHALL be measured, and the evaluation result SHALL be identical to the harness's output before the parameter existed.

#### Scenario: Concatenated baseline integration
- **WHEN** the caller selects `concat` integration
- **THEN** the harness converts methylation to M-values, standardises all selected layers, and concatenates them into the outcome matrix
- **AND** the result metadata identifies `concat` as a baseline rather than a constructed latent space

#### Scenario: SNF integration
- **WHEN** the caller selects `snf` integration
- **THEN** the harness converts methylation to M-values and creates the latent space from SNF fusion of the selected layers and spectral embedding

#### Scenario: PLS integration
- **WHEN** the caller selects `pls` integration
- **THEN** the harness converts methylation to M-values, standardises the selected layers, fits PLS-DA conditioned on the stage label, and returns the PLS X-score matrix as the latent space
- **AND** the number of latent variables is selected by the double nested cross-validation (modal LV across repeats, parsimony tie-break) to secure a stable, non-overfitted molecular space
- **AND** the result metadata records the selected number of latent variables and the cross-validation parameters

#### Scenario: PLS integration is infeasible
- **WHEN** the caller selects `pls` integration but the sample provides too few observations per stage for the cross-validation
- **THEN** the harness raises a clear validation error

#### Scenario: Unsupported integration method
- **WHEN** the caller selects an unsupported integration method
- **THEN** the harness raises a clear validation error

#### Scenario: A block subset is measured
- **WHEN** the caller selects the layers methylation and expression
- **THEN** the outcome matrix, the latent space, every realized-geometry checkpoint's joint scope, and every attribution feature record contain methylation and expression features only
- **AND** realized geometry reports no proteomics per-block scope
- **AND** the result metadata records the selected layers in canonical order

#### Scenario: Absent layer selection is byte-identical
- **WHEN** the evaluation parameters omit the layer selection
- **THEN** the evaluation result equals the result produced before the selection existed, for the same dataset and parameters

#### Scenario: Invalid layer selection is rejected
- **WHEN** the layer selection is empty, names an unknown layer, or repeats a layer
- **THEN** the harness raises a clear validation error naming the allowed layers
