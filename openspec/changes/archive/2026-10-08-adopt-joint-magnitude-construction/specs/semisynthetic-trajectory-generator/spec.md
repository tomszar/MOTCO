# Spec Delta

## MODIFIED Requirements

### Requirement: Trajectory modes are feature-set surgery on methylation differential indicators
The generator SHALL define `none`, `translation`, `magnitude`, `orientation`, and `shape` as operations on group B's **per-stage methylation** differential-feature indicators only. Group A inherits a baseline set of per-stage methylation indicators (which need not form a continuous/straight trajectory). For both groups, the gene-expression and proteomics differential indicators SHALL be **derived from the (group-specific) methylation indicators** through the cached CpG→gene→protein incidence maps — the surgery never touches expression, proteomics, or the latent space directly. This keeps the simulated differences biologically grounded (methylation drives expression drives protein) and keeps the data realistic rather than tailored to MOTCO. The one surgery that acts on effect sizes rather than indicators is `magnitude`, whose per-omic δ scaling is selected by `magnitude_kind`.

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
- **WHEN** `trajectory_mode` is `magnitude` and `magnitude_kind` is `all` or `extremes`
- **THEN** group B uses the same per-stage methylation indicators as group A but with a scaled methylation effect size `δ_methyl_B = (1 + e)·δ_methyl` (at every stage for `all`, through the endpoint indicators for `extremes`), leaving `δ_expr` and `δ_protein` at group A's values — a size/`delta` change that is exactly size-pure within each omic block but not in the joint space

#### Scenario: Joint magnitude mode scales every omic's effect together
- **WHEN** `trajectory_mode` is `magnitude` and `magnitude_kind` is `joint`
- **THEN** group B uses the same per-stage methylation indicators as group A with every per-omic effect size scaled by the same factor, `δ_methyl_B = (1 + e)·δ_methyl`, `δ_expr_B = (1 + e)·δ_expr`, `δ_protein_B = (1 + e)·δ_protein`, so group B's native-space stage-mean trajectory is exactly `(1 + e)` times group A's in every omic block — a size/`delta` change that is size-pure in the joint space as well

#### Scenario: Orientation mode relocates stage-changing sites consistently across stages
- **WHEN** `trajectory_mode` is `orientation`
- **THEN** a fraction `e` of group A's stage-changing methylation sites are relocated to different CpGs using a single relocation applied identically to every stage, so the per-stage on/off pattern is preserved on different feature axes (a rotation: orientation changes, with size and shape preserved in the linear limit)

#### Scenario: Shape mode perturbs a single interior stage
- **WHEN** `trajectory_mode` is `shape` and at least three stages are available
- **THEN** group B perturbs a single interior stage relative to group A — either by relocating a fraction `e` of that stage's methylation sites (`relocate`) or by scaling that stage's methylation effect (`magnitude`) — bending one interior vertex of the trajectory (a shape change, which may co-move size)

#### Scenario: Shape mode rejects fewer than three stages
- **WHEN** `trajectory_mode` is `shape` and fewer than three stages are available
- **THEN** the generator raises a clear validation error

### Requirement: Magnitude mode supports an extreme-stage variant
The generator SHALL provide a `magnitude_kind` option with three selectable values: `all` (the default) scales group B's methylation effect at **all** stages; `extremes` scales it only at the **extreme** stages (the first and last); `joint` scales **every omic's** effect size — methylation, expression, and proteomics — by the same factor `1 + e` at all stages. The option is backward-compatible: the default reproduces the existing all-stage, methylation-only behavior byte for byte at every seed, and adding `joint` MUST NOT change the dataset, truth, or parameter signature of any configuration that does not select it. An unknown value SHALL be rejected with a clear validation error naming the allowed values. `joint` is a production value: it SHALL be selectable through the generator parameters, the study configuration loader, and the `motco simulate` command line.

#### Scenario: All-stages magnitude is the default
- **WHEN** `trajectory_mode` is `magnitude` and `magnitude_kind` is unset
- **THEN** group B's methylation effect is scaled at every stage (the existing behavior)

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

#### Scenario: Configurations that do not name a kind are unchanged
- **WHEN** a study configuration or generator call omits `magnitude_kind`
- **THEN** its datasets, truth metadata, and parameter signatures are byte-identical to those produced before `joint` existed

#### Scenario: Magnitude variant is recorded as truth
- **WHEN** a `magnitude` dataset is generated
- **THEN** truth metadata records which `magnitude_kind` was used and, for `joint`, the common scale factor applied to each of the three per-omic effect sizes
