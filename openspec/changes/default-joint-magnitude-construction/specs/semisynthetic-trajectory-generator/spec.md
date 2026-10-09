# Spec Delta

## REMOVED Requirements

### Requirement: Magnitude mode supports an extreme-stage variant
**Reason**: The requirement made `all` the default and promised that configurations omitting the kind are unchanged, and its title describes only one of three values. Both are superseded by the `joint` default.
**Migration**: Replaced by "Magnitude kind selects the construction, joint by default" below. Callers wanting the former default pass `magnitude_kind="all"` (`--magnitude-kind all`). Committed configurations are pinned per "Historical configurations pin the pre-default magnitude construction".

## ADDED Requirements

### Requirement: Magnitude kind selects the construction, joint by default
The generator SHALL provide a `magnitude_kind` option with three selectable values: `joint` (the default) scales **every omic's** effect size (methylation, expression and proteomics) by the same factor `1 + e` at all stages; `all` scales group B's methylation effect alone at **all** stages; `extremes` scales the methylation effect only at the **extreme** stages (the first and last). `all` and `extremes` SHALL reproduce their existing methylation-only behavior byte for byte at every seed when selected explicitly. An unknown value SHALL be rejected with a clear validation error naming the allowed values. Every value SHALL be selectable through the generator parameters, the study configuration loader, and the `motco simulate` command line, and the command line's default SHALL be the generator's default. `extremes` and the shape mode's `magnitude` kind remain methylation-only scalings; the joint default does not extend to them.

#### Scenario: Joint magnitude is the default
- **WHEN** `trajectory_mode` is `magnitude` and `magnitude_kind` is unset
- **THEN** group B's methylation, expression, and proteomics effect sizes are each `(1 + e)` times group A's, and truth metadata records `magnitude_kind` as `joint`

#### Scenario: All-stages methylation-only magnitude is selectable
- **WHEN** `trajectory_mode` is `magnitude` and `magnitude_kind` is `all`
- **THEN** group B's methylation effect is scaled at every stage with expression and proteomics at group A's values, and the dataset and truth are identical to those the generator produced for an unset `magnitude_kind` before the default changed, at the same seed

#### Scenario: Extreme-stage magnitude scales only the endpoints
- **WHEN** `trajectory_mode` is `magnitude` and `magnitude_kind` is `extremes`
- **THEN** group B's methylation effect is scaled only at the first and last stages, leaving interior stages at the baseline effect

#### Scenario: Joint magnitude scales all three per-omic effects
- **WHEN** `trajectory_mode` is `magnitude` and `magnitude_kind` is `joint` with effect size `e`
- **THEN** group B's methylation, expression, and proteomics effect sizes are each `(1 + e)` times group A's, and group B's methylation indicators are identical to group A's

#### Scenario: Joint magnitude is exactly null at zero effect
- **WHEN** `trajectory_mode` is `magnitude`, `magnitude_kind` is `joint`, and `group_effect_size` is 0
- **THEN** the generated dataset and truth are identical to the `none` mode's at the same seed

#### Scenario: Joint magnitude is selectable from a study configuration
- **WHEN** a study configuration sets `generator.magnitude_kind` to `joint`
- **THEN** the configuration loads, enumerates, and generates with the joint construction, and a configuration naming any value outside `all`, `extremes`, `joint` is still refused at load time

#### Scenario: Command-line default follows the generator
- **WHEN** `motco simulate` is run in `magnitude` mode without `--magnitude-kind`
- **THEN** the written truth records `magnitude_kind` as `joint`, and passing `--magnitude-kind all` reproduces the methylation-only construction

#### Scenario: Magnitude variant is recorded as truth
- **WHEN** a `magnitude` dataset is generated
- **THEN** truth metadata records which `magnitude_kind` was used and, for `joint`, the common scale factor applied to each of the three per-omic effect sizes

### Requirement: Historical configurations pin the pre-default magnitude construction
Every committed study configuration that predates the `joint` default and does not already name a magnitude kind SHALL name `magnitude_kind` `all` explicitly in its generator block, so that changing the default alters none of its enumerated cells' parameter signatures, cell identifiers, matched seeds, or generated datasets. Every committed diagnostic that constructs generator parameters directly, and whose committed results were produced under the former default, SHALL likewise select `all` explicitly. The pins record historical behavior and MUST be documented as such, so that new configurations do not copy them.

#### Scenario: Historical config signatures are unchanged by the default flip
- **WHEN** each historical study configuration is loaded and enumerated after the default changes
- **THEN** every enumerated cell's parameter signature equals the signature recorded for that cell before the change

#### Scenario: Historical diagnostics regenerate their committed data
- **WHEN** a pinned historical diagnostic is re-run with its committed command after the default changes
- **THEN** it generates the same datasets it generated before the change

#### Scenario: New configuration without a kind gets the joint construction
- **WHEN** a study configuration that does not set `generator.magnitude_kind` requests the magnitude mode
- **THEN** it resolves to the `joint` construction
