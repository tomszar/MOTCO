## ADDED Requirements

### Requirement: Magnitude mode supports an extreme-stage variant
The generator SHALL provide a `magnitude_kind` option that selects whether the
`magnitude` mode scales group B's methylation effect at **all** stages (the
default) or only at the **extreme** stages (the first and last). The variant is
backward-compatible: the default reproduces the existing all-stage behavior.

#### Scenario: All-stages magnitude is the default
- **WHEN** `trajectory_mode` is `magnitude` and `magnitude_kind` is unset
- **THEN** group B's methylation effect is scaled at every stage (the existing behavior)

#### Scenario: Extreme-stage magnitude scales only the endpoints
- **WHEN** `trajectory_mode` is `magnitude` and `magnitude_kind` is `extremes`
- **THEN** group B's methylation effect is scaled only at the first and last stages, leaving interior stages at the baseline effect

#### Scenario: Magnitude variant is recorded as truth
- **WHEN** a `magnitude` dataset is generated
- **THEN** truth metadata records which `magnitude_kind` was used
