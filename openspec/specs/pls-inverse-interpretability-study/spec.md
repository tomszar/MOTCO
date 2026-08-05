# pls-inverse-interpretability-study Specification

## Purpose
TBD - created by archiving change add-pls-inverse-interpretability-study. Update Purpose after archive.
## Requirements
### Requirement: Exact-null feature datasets

The study SHALL generate deterministic Gaussian feature data for two and three stages and SHALL construct Groups A and B as exact paired duplicates before any latent intervention.

#### Scenario: Baseline groups are identical

- **WHEN** a study dataset is generated for a fixed seed and stage count
- **THEN** corresponding Group A and Group B feature rows, stage centroids, and initial trajectory geometry are identical

### Requirement: Fixed two-component PLS-DA representation

The study SHALL fit one pooled two-component PLS-DA model using stage as the response and SHALL reuse that frozen fitted model for all transformations and reconstructions within a dataset.

#### Scenario: Intervention does not refit PLS

- **WHEN** any latent trajectory intervention is applied
- **THEN** the modified scores and reconstructed features use the same fitted model as the unmodified scores

### Requirement: Controlled latent trajectory interventions

The study SHALL support magnitude and orientation interventions for two-stage trajectories and magnitude, orientation, and shape-bend interventions for three-stage trajectories. Interventions SHALL modify only Group B stage centroids and SHALL preserve sample residuals around each original group-stage centroid.

#### Scenario: Magnitude intervention

- **WHEN** magnitude is applied with scale factor `c`
- **THEN** Group B stage centroids are scaled about their trajectory centroid by `c`, with the trajectory centroid and orientation unchanged in PLS space

#### Scenario: Orientation intervention

- **WHEN** orientation is applied with angle `theta`
- **THEN** Group B stage centroids are rigidly rotated about their trajectory centroid in the LV1-LV2 plane, preserving all latent pairwise distances

#### Scenario: Three-stage shape intervention

- **WHEN** shape is applied to a three-stage trajectory
- **THEN** the endpoint centroids remain fixed and the middle centroid moves in a direction perpendicular to the endpoint axis

#### Scenario: Within-stage residuals are preserved

- **WHEN** any intervention moves a Group B stage centroid
- **THEN** every sample retains its original score residual relative to that stage centroid

### Requirement: Additive feature reconstruction

The study SHALL reconstruct the implied Group B feature change as the difference between the fitted PLS inverse of modified and original scores and SHALL add that difference to the original Group B features. Group A features MUST remain unchanged.

#### Scenario: Original feature residual is retained

- **WHEN** modified Group B scores are reconstructed
- **THEN** the result equals the original Group B matrix plus the PLS-represented intervention rather than a replacement by the low-rank PLS reconstruction

#### Scenario: Score displacement round trip is measured

- **WHEN** reconstructed features are transformed through the frozen PLS model
- **THEN** the study reports the discrepancy between the requested and recovered score displacements

### Requirement: Latent and feature geometry comparison

The study SHALL summarize trajectory magnitude, orientation, and shape in PLS space and original feature space for every applicable stage-count and intervention combination, with shape marked unavailable for two stages.

#### Scenario: Two-stage result matrix

- **WHEN** the two-stage study runs
- **THEN** results contain magnitude and orientation interventions with latent and feature-space magnitude and angle summaries and no finite shape comparison

#### Scenario: Three-stage result matrix

- **WHEN** the three-stage study runs
- **THEN** results contain magnitude, orientation, and shape interventions with latent and feature-space magnitude, angle, and shape summaries

### Requirement: Feature-change interpretability diagnostics

The study SHALL report per-feature signed and absolute changes, associated PLS loading information, aggregate change concentration, and the metric induced by the PLS X loadings.

#### Scenario: Feature changes are ranked

- **WHEN** an intervention is reconstructed
- **THEN** a tidy feature table ranks features by absolute implied change and includes their component loadings

#### Scenario: Induced metric is reported

- **WHEN** the fitted PLS model is summarized
- **THEN** the study reports `G = P^T P` and diagnostics of its anisotropy

### Requirement: Reproducible study driver

The system SHALL provide a driver that runs all required stage-count and intervention cells from explicit parameters and writes machine-readable results plus a compact Markdown summary.

#### Scenario: Default study run

- **WHEN** the driver is invoked with defaults
- **THEN** it runs two-stage magnitude/orientation and three-stage magnitude/orientation/shape cells with a fixed seed and records all effective parameters

