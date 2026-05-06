## Purpose

Generate MOTCO-ready semi-synthetic trajectory datasets from InterSIM outputs using the clusters-as-stages assumption, reproducible group assignment, trajectory effect injection, and explicit truth metadata.

## Requirements

### Requirement: Generator creates MOTCO-ready trajectory datasets from InterSIM outputs
MOTCO SHALL provide a semi-synthetic trajectory generator that consumes an `InterSIMResult` and returns aligned omics matrices, sample metadata, and truth metadata.

#### Scenario: Successful transformation from InterSIMResult
- **WHEN** a caller provides an aligned `InterSIMResult` and valid generator parameters
- **THEN** the generator returns methylation, gene expression, and proteomics matrices aligned to sample metadata

#### Scenario: Sample metadata contains trajectory design columns
- **WHEN** generation succeeds
- **THEN** sample metadata contains `sample_id`, `group`, `stage`, and `cluster` columns

#### Scenario: Truth metadata records generator parameters
- **WHEN** generation succeeds
- **THEN** truth metadata records trajectory mode, group effect size, affected features, stage mapping, group assignment seed, and the clusters-as-stages assumption

### Requirement: InterSIM clusters are mapped to ordered trajectory stages
The generator SHALL map InterSIM cluster labels to ordered integer stages using a deterministic clusters-as-stages assumption.

#### Scenario: Cluster labels are converted to stages
- **WHEN** InterSIM cluster labels are present
- **THEN** the generator maps sorted unique cluster labels to integer stages starting at 0

#### Scenario: Original cluster labels are preserved
- **WHEN** cluster labels are mapped to stages
- **THEN** sample metadata preserves the original InterSIM cluster label in the `cluster` column

#### Scenario: Stage mapping is recorded
- **WHEN** generation succeeds
- **THEN** truth metadata includes the mapping from original cluster labels to generated stage labels

### Requirement: Groups are assigned reproducibly within stages
The generator SHALL assign comparison group labels reproducibly within each stage according to configured group balance.

#### Scenario: Two groups are assigned within every stage
- **WHEN** each stage has enough samples for two groups
- **THEN** the generator assigns group labels within each stage according to the configured group ratio

#### Scenario: Same seed gives same group labels
- **WHEN** the same InterSIM output, generator parameters, and seed are used twice
- **THEN** the generated group labels are identical

#### Scenario: Insufficient stage size is rejected
- **WHEN** any stage has too few samples to assign both comparison groups
- **THEN** the generator raises a clear validation error

### Requirement: Generator supports initial trajectory truth modes
The generator SHALL support `none`, `translation`, `magnitude`, `orientation`, and `shape` trajectory modes.

#### Scenario: None mode preserves InterSIM omics values
- **WHEN** `trajectory_mode` is `none`
- **THEN** the generator does not add group-specific shifts to any omics matrix

#### Scenario: Zero effect size preserves InterSIM omics values
- **WHEN** `group_effect_size` is 0
- **THEN** the generator does not add group-specific shifts to any omics matrix

#### Scenario: Translation mode applies stage-invariant group shift
- **WHEN** `trajectory_mode` is `translation` and `group_effect_size` is greater than 0
- **THEN** the generator adds the same group-specific feature shift to affected group samples in all stages

#### Scenario: Magnitude mode applies stage-proportional group shift
- **WHEN** `trajectory_mode` is `magnitude` and `group_effect_size` is greater than 0
- **THEN** the generator adds stage-proportional group-specific feature shifts to affected group samples

#### Scenario: Orientation mode applies off-axis stage-proportional group shift
- **WHEN** `trajectory_mode` is `orientation` and `group_effect_size` is greater than 0
- **THEN** the generator adds stage-proportional group-specific feature shifts along an off-axis direction

#### Scenario: Shape mode applies non-monotone stage-specific group shift
- **WHEN** `trajectory_mode` is `shape` and at least three stages are available
- **THEN** the generator adds non-monotone stage-specific group shifts to affected group samples

#### Scenario: Shape mode rejects fewer than three stages
- **WHEN** `trajectory_mode` is `shape` and fewer than three stages are available
- **THEN** the generator raises a clear validation error

### Requirement: Affected features are selected deterministically
The generator SHALL select affected features per omics layer deterministically from explicit feature lists or a configured affected-feature proportion.

#### Scenario: Explicit affected features are honored
- **WHEN** a caller provides explicit affected feature names for an omics layer
- **THEN** the generator uses exactly those features for injected shifts in that layer

#### Scenario: Proportion-based affected features are reproducible
- **WHEN** a caller provides an affected-feature proportion and seed
- **THEN** the generator selects the same affected features for repeated runs with the same inputs

#### Scenario: Affected features are recorded
- **WHEN** generation succeeds
- **THEN** truth metadata records affected feature names for every omics layer

### Requirement: Generator can invoke InterSIM as a convenience path
MOTCO SHALL provide a convenience API that invokes the existing InterSIM bridge and then generates a semi-synthetic trajectory dataset.

#### Scenario: InterSIM-backed generation succeeds
- **WHEN** InterSIM is available and a caller provides valid InterSIM and generator parameters
- **THEN** the convenience API returns a semi-synthetic trajectory dataset

#### Scenario: InterSIM failure propagates clearly
- **WHEN** the InterSIM bridge fails during convenience generation
- **THEN** the convenience API raises the bridge error without obscuring the InterSIM failure details
