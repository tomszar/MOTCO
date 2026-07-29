## ADDED Requirements

### Requirement: Dominant-specificity study runs through a selectable latent space

The dominant-specificity instrumentation SHALL accept a latent-space selection (`integration_method` with optional `integration_params`) and run the full semi-synthetic generator → integration → `estimate_difference` → RRPP rejection-rate study in that latent space. The selection MUST apply to both the per-statistic RRPP rejection-rate evaluation and the group-in-stage projection, and MUST default to the `concat` baseline so existing callers are unchanged.

#### Scenario: Study runs in the PLS latent space
- **WHEN** a caller selects `pls` integration for `evaluate_mode_specificity`
- **THEN** each replicate is measured in the PLS molecular latent space and the per-statistic (`delta`/`angle`/`shape`) rejection rates and group-in-stage fraction are reported for that space

#### Scenario: Latent space applies to the group-in-stage projection
- **WHEN** a latent space is selected
- **THEN** the group-in-stage fraction is computed in the same selected latent space as the rejection-rate evaluation

#### Scenario: Default is the concat baseline
- **WHEN** no integration method is specified
- **THEN** the study runs through the `concat` baseline, reproducing the prior behavior

### Requirement: Cross-latent-space comparison driver

The system SHALL provide a driver that runs each trajectory mode (`magnitude`, `orientation`, `shape`, `none`) through the `concat`, `snf`, and `pls` latent spaces on matched seeds and effect size, and reports the per-statistic rejection-rate table and group-in-stage fraction per latent space.

#### Scenario: Matched comparison across latent spaces
- **WHEN** the driver is run
- **THEN** the only varied factor across the reported tables is the latent space (seeds, effect size, stages, and permutation count are held identical), so differences in rejection rates are attributable to the latent space
