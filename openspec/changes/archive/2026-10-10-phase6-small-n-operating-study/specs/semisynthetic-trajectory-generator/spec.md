# Spec Delta

## MODIFIED Requirements

### Requirement: Groups are assigned reproducibly within stages
The generator SHALL assign comparison group labels reproducibly within each stage according to configured group balance. Group balance SHALL be configurable in exactly one of two ways. The proportional way uses a total sample count, optional per-stage proportions, and one group ratio applied in every stage. The explicit way uses a group × stage sample-size table, one sequence of per-stage counts per group, ordered as the group labels and the stages. When the explicit table is given, every group-stage cell SHALL contain exactly the stated number of samples. The total sample count and stage counts SHALL be derived from the table. The generator SHALL reject a configuration that sets the explicit table together with any proportional setting that differs from its default. When the explicit table is absent, generation SHALL be identical to the generator's behavior before the table existed, at every seed.

#### Scenario: Two groups are assigned within every stage
- **WHEN** each stage has enough samples for two groups and no explicit size table is given
- **THEN** the generator assigns group labels within each stage according to the configured group ratio

#### Scenario: Same seed gives same group labels
- **WHEN** the same generator parameters and seed are used twice
- **THEN** the generated group labels are identical

#### Scenario: Insufficient stage size is rejected
- **WHEN** any stage has too few samples to assign both comparison groups
- **THEN** the generator raises a clear validation error

#### Scenario: Explicit group × stage sizes are realized exactly
- **WHEN** the caller supplies the size table `((11, 10, 28), (9, 10, 12))` with three stages
- **THEN** the generated metadata contains exactly 11, 10, and 28 samples of the first group and 9, 10, and 12 samples of the second group at stages 0, 1, and 2, and 80 samples in total
- **AND** truth metadata records the realized group × stage sizes

#### Scenario: Explicit table conflicts with proportional settings
- **WHEN** the caller supplies the size table together with a non-default `n_samples`, `stage_sample_prop`, or `group_ratio`
- **THEN** the generator raises a clear validation error naming the conflicting settings

#### Scenario: Malformed explicit table is rejected
- **WHEN** the size table has a number of rows other than two, a row length other than the number of stages, or any cell below one sample
- **THEN** the generator raises a clear validation error

#### Scenario: Absent table is byte-identical
- **WHEN** generator parameters omit the size table
- **THEN** the generated dataset, truth metadata, and parameter signature equal those produced before the table existed, at the same seed
