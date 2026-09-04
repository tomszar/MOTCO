# semisynthetic-trajectory-generator Specification

## Purpose

Generate MOTCO-ready semi-synthetic trajectory datasets from InterSIM outputs using the clusters-as-stages assumption, reproducible group assignment, trajectory effect injection, and explicit truth metadata.

## Requirements

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

### Requirement: Pool-limited surgeries apply an explicit censoring policy
The generator SHALL apply an explicit, configurable censoring policy to every surgery whose size is
limited by a destination or candidate pool (`orientation` relocation, `translation` extra set,
`shape` with `shape_kind='relocate'`). The policy SHALL default to failing loudly: when the
requested surgery size exceeds the available pool, generation raises a validation error naming the
mode, the requested size, and the available pool size. An explicit opt-in policy value SHALL
preserve the previous clamping behavior (realize as much of the surgery as the pool allows). The
generator MUST NOT silently clamp under the default policy.

#### Scenario: Default policy fails loudly on a censored surgery
- **WHEN** a pool-limited surgery's requested size exceeds its available pool and no censoring
  policy is explicitly configured
- **THEN** generation raises a clear validation error identifying the trajectory mode, the
  requested surgery size, and the available pool size, and no dataset is returned

#### Scenario: Opt-in clamping preserves the previous behavior
- **WHEN** the censoring policy is explicitly set to clamp and a pool-limited surgery's requested
  size exceeds its available pool
- **THEN** the generator realizes the largest surgery the pool allows and generation succeeds,
  matching the pre-policy behavior replicate for replicate at the same seed

#### Scenario: Uncensored surgeries are unaffected by the policy
- **WHEN** a pool-limited surgery's requested size does not exceed its available pool
- **THEN** the realized surgery equals the requested size under either policy value, and the
  sampled dataset at a given seed is identical to the pre-policy generator's output

### Requirement: Truth metadata makes requested-vs-realized surgery explicit
For every pool-limited surgery, truth metadata SHALL record the nominal (requested) surgery size
beside the realized surgery size, and a flag stating whether the surgery was censored (realized
size smaller than nominal). This makes the requested-effect → realized-surgery relationship
auditable per record, without regenerating data.

#### Scenario: Nominal and realized sizes are recorded together
- **WHEN** a dataset with a pool-limited surgery is generated successfully
- **THEN** truth metadata contains the nominal surgery size, the realized surgery size, and a
  censored flag that is true exactly when realized < nominal

#### Scenario: Clamped generation is identifiable downstream
- **WHEN** a dataset is generated under the clamping policy and the clamp binds
- **THEN** the truth metadata's censored flag is true, so downstream summaries can identify the
  record without access to the generator

### Requirement: Baseline stage-program continuity is a generator axis
The generator SHALL expose a baseline continuity parameter ρ ∈ [0, 1) controlling how strongly
group A's per-stage methylation differential indicators persist across adjacent stages. Each CpG's
indicator sequence across stages SHALL follow a stationary first-order Markov chain whose per-stage
marginal is Bernoulli(`p_dmp`) at every ρ (first stage ~ Bernoulli(`p_dmp`); stay-differential
probability `p_dmp + ρ·(1 − p_dmp)`; become-differential probability `p_dmp·(1 − ρ)`), so that the
cross-stage indicator correlation is ρ^|t−s| while per-stage indicator counts, the per-omic effect
sizes, the cross-omic coupling derivation, and the meaning of `p_dmp` are unchanged along the axis.
At ρ = 0 the generator SHALL reproduce the independent per-stage draws of the pre-change generator
byte-identically at the same seed. Values outside [0, 1) SHALL be rejected with a clear validation
error, and truth metadata SHALL record the continuity value.

#### Scenario: Marginals are preserved along the axis
- **WHEN** datasets are generated at any ρ ∈ [0, 1)
- **THEN** each stage's methylation indicators are marginally Bernoulli(`p_dmp`), and the expected
  per-stage differential count does not depend on ρ

#### Scenario: Zero continuity reproduces the independent baseline exactly
- **WHEN** a dataset is generated with ρ = 0 and any seed
- **THEN** the baseline indicators, all derived indicators, and the sampled dataset are identical
  to the pre-change generator's output at that seed

#### Scenario: Continuity induces trending stage configurations
- **WHEN** ρ > 0
- **THEN** adjacent stages share differential programs with correlation ρ, so expected squared
  stage-mean distances grow with stage separation (proportional to 1 − ρ^|t−s|) instead of being
  equal for all pairs

#### Scenario: Continuity is recorded as truth
- **WHEN** generation succeeds
- **THEN** truth metadata records the baseline continuity value beside the other generator
  parameters

#### Scenario: Invalid continuity is rejected
- **WHEN** a caller passes a continuity value below 0 or at/above 1
- **THEN** generation raises a validation error naming the parameter and the allowed range

### Requirement: Expected surgery headroom accounts for baseline continuity
The generator's analytic expected-headroom computation for pool-limited surgeries SHALL use the
continuity-adjusted probability that a CpG is differential in at least one stage,
`1 − (1 − p_dmp)·(1 − p_dmp·(1 − ρ))^(n_stages − 1)`, which reduces to the independence union
probability at ρ = 0. Expected destination pools and saturating effects SHALL therefore reflect
that higher continuity shrinks the expected stage-active union (stage programs overlap more).

#### Scenario: Headroom at zero continuity is unchanged
- **WHEN** expected headroom is computed at ρ = 0
- **THEN** the expected pools and saturating effects equal the pre-change independence-based values

#### Scenario: Headroom tracks the continuity-adjusted union
- **WHEN** expected headroom is computed at ρ > 0 for an orientation or shape-relocate surgery
- **THEN** the expected stage-active fraction uses the continuity-adjusted union probability, and
  the reported saturating effect is correspondingly no smaller than at ρ = 0 for the same
  `p_dmp` and `n_stages`
