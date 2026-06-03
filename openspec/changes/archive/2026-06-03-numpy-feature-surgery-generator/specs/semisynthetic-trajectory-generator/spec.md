## MODIFIED Requirements

### Requirement: Groups are assigned reproducibly within stages
The generator SHALL assign comparison group labels reproducibly within each stage according to configured group balance.

#### Scenario: Two groups are assigned within every stage
- **WHEN** each stage has enough samples for two groups
- **THEN** the generator assigns group labels within each stage according to the configured group ratio

#### Scenario: Same seed gives same group labels
- **WHEN** the same generator parameters and seed are used twice
- **THEN** the generated group labels are identical

#### Scenario: Insufficient stage size is rejected
- **WHEN** any stage has too few samples to assign both comparison groups
- **THEN** the generator raises a clear validation error

## REMOVED Requirements

### Requirement: Generator creates MOTCO-ready trajectory datasets from InterSIM outputs
**Reason**: Generation no longer consumes an `InterSIMResult` or an R subprocess; it is replaced by the numpy-generator path below.

### Requirement: InterSIM clusters are mapped to ordered trajectory stages
**Reason**: Stages are produced directly by the numpy generator's per-stage indicators; there is no InterSIM cluster-to-stage mapping step.

### Requirement: Generator supports initial trajectory truth modes
**Reason**: The trajectory modes are redefined as methylation-only feature-set surgery (see the new requirement below), superseding the random-direction truth modes.

### Requirement: Affected features are selected deterministically
**Reason**: Group effects are no longer injected along randomly selected feature sets; they are deterministic transforms of the per-stage methylation indicators.

### Requirement: Generator can invoke InterSIM as a convenience path
**Reason**: The convenience path no longer invokes InterSIM/R; it is replaced by the numpy-generator convenience requirement below.

## ADDED Requirements

### Requirement: Generator creates MOTCO-ready trajectory datasets from the numpy generator
MOTCO SHALL provide a semi-synthetic trajectory generator that builds datasets on top of the numpy omics generator and returns aligned omics matrices, sample metadata, and truth metadata, without an `InterSIMResult` or an R subprocess at runtime.

#### Scenario: Successful generation from the numpy generator
- **WHEN** a caller provides valid generator parameters
- **THEN** the generator returns methylation, gene expression, and proteomics matrices aligned to sample metadata, produced without invoking R

#### Scenario: Sample metadata contains trajectory design columns
- **WHEN** generation succeeds
- **THEN** sample metadata contains `sample_id`, `group`, `stage`, and `cluster` columns

#### Scenario: Truth metadata records generator parameters
- **WHEN** generation succeeds
- **THEN** truth metadata records trajectory mode, group effect size, per-stage/group differential indicators, per-omic effect sizes, and the group-assignment seed

### Requirement: Trajectory modes are feature-set surgery on methylation differential indicators
The generator SHALL define `none`, `translation`, `magnitude`, `orientation`, and `shape` as operations on group B's **per-stage methylation** differential-feature indicators only. Group A inherits a baseline set of per-stage methylation indicators (which need not form a continuous/straight trajectory). For both groups, the gene-expression and proteomics differential indicators SHALL be **derived from the (group-specific) methylation indicators** through the cached CpG→gene→protein incidence maps — the surgery never touches expression, proteomics, or the latent space directly. This keeps the simulated differences biologically grounded (methylation drives expression drives protein) and keeps the data realistic rather than tailored to MOTCO.

#### Scenario: Group B expression and protein indicators are derived from its methylation
- **WHEN** any non-null mode transforms group B's methylation indicators
- **THEN** group B's expression and proteomics differential indicators are re-derived from group B's methylation indicators via the incidence maps (not manipulated independently)

#### Scenario: None mode gives identical group trajectories
- **WHEN** `trajectory_mode` is `none` (or `group_effect_size` is 0)
- **THEN** group B uses the same methylation indicators and effects as group A, so the groups share an identical trajectory

#### Scenario: Translation mode adds an extra constant differential set
- **WHEN** `trajectory_mode` is `translation`
- **THEN** group B keeps group A's stage-changing methylation sites unchanged and additionally marks an extra set `U` of methylation sites — whose mapped genes are absent from the stage program — as differential at every group-B stage (and at none of group A's), producing a constant group offset that leaves the size, orientation, and shape statistics unchanged

#### Scenario: Magnitude mode scales the methylation effect
- **WHEN** `trajectory_mode` is `magnitude`
- **THEN** group B uses the same per-stage methylation indicators as group A but with a scaled methylation effect size `δ_methyl_B = (1 + e)·δ_methyl`, scaling every methylation transition (a size/`delta` change)

#### Scenario: Orientation mode relocates stage-changing sites consistently across stages
- **WHEN** `trajectory_mode` is `orientation`
- **THEN** a fraction `e` of group A's stage-changing methylation sites are relocated to different CpGs using a single relocation applied identically to every stage, so the per-stage on/off pattern is preserved on different feature axes (a rotation: orientation changes, with size and shape preserved in the linear limit)

#### Scenario: Shape mode perturbs a single interior stage
- **WHEN** `trajectory_mode` is `shape` and at least three stages are available
- **THEN** group B perturbs a single interior stage relative to group A — either by relocating a fraction `e` of that stage's methylation sites (`relocate`) or by scaling that stage's methylation effect (`magnitude`) — bending one interior vertex of the trajectory (a shape change, which may co-move size)

#### Scenario: Shape mode rejects fewer than three stages
- **WHEN** `trajectory_mode` is `shape` and fewer than three stages are available
- **THEN** the generator raises a clear validation error

### Requirement: Baseline indicators and the group transform are deterministic
The generator SHALL construct group A's baseline per-stage methylation indicators and group B's transform deterministically from the seed and parameters, recording both in truth metadata.

#### Scenario: Same seed gives the same indicators
- **WHEN** the same parameters and seed are used twice
- **THEN** the per-stage/group differential indicators are identical across runs

#### Scenario: Indicators are recorded as truth
- **WHEN** generation succeeds
- **THEN** truth metadata records the per-stage methylation (and derived expression/protein) indicators for both groups, the per-omic effect sizes, and any extra translation set

### Requirement: Generator provides a single-call convenience path
MOTCO SHALL provide a convenience API that generates a semi-synthetic trajectory dataset end to end from parameters, using the numpy generator and cached reference data.

#### Scenario: End-to-end generation succeeds without R
- **WHEN** a caller provides valid parameters
- **THEN** the convenience API returns a semi-synthetic trajectory dataset using only cached reference data, with no R invocation

### Requirement: Generator emits per-stage/group differential-indicator truth for characterization
The generator SHALL expose the differential-feature indicators for each stage and group so a consumer can characterize how MOTCO responds to an injected mode (which statistics move, and how much cross-talk there is).

#### Scenario: Indicator truth is consumable downstream
- **WHEN** a dataset is generated
- **THEN** the per-stage/group indicators are available in the dataset's truth structure for the showcase and study to characterize mode-to-statistic response (a descriptive specificity matrix, not a pass/fail gate)
