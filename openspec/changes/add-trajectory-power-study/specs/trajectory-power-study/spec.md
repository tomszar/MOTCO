## ADDED Requirements

### Requirement: Study is defined by a declarative configuration

MOTCO SHALL provide a declarative, file-based study configuration that fully determines an enumerated grid of simulation cells, so that a study is reproducible from its config alone.

The configuration MUST capture baseline InterSIM/generator/evaluation parameters, the set of trajectory modes, the set of effect sizes, the one-factor-at-a-time axes with their values, the per-cell replicate count, the base seed, and the pre-specified acceptance targets.

#### Scenario: Config enumerates a deterministic grid

- **WHEN** a study configuration is loaded and enumerated
- **THEN** it produces a `SimulationGrid` of Type I and power cells with stable, deterministic cell identifiers
- **AND** enumerating the same configuration twice yields identical cell identifiers and parameter signatures

#### Scenario: Config records negative-control modes

- **WHEN** a study configuration is enumerated
- **THEN** the resulting Type I cells include both the `none` (no group effect) and `translation` (location-only group effect) trajectory modes as negative controls

#### Scenario: Invalid configuration is rejected

- **WHEN** a study configuration omits a required field or specifies an unknown trajectory mode, a negative replicate count, or an unknown axis namespace
- **THEN** loading the configuration raises a clear validation error identifying the offending field

### Requirement: Study executes as resumable per-shard work units

MOTCO SHALL execute an enumerated study as `(cell, replicate)` work units partitioned across a fixed number of shards, so that the study can run as parallel cluster tasks without coordination.

Each shard MUST persist its own JSONL output file and MUST be independently resumable using the existing parameter-signature guard, so re-running a shard skips already-completed replicates and never appends duplicates.

#### Scenario: Work is partitioned deterministically across shards

- **WHEN** a study is run with `n_shards = N` and a given shard index `i`
- **THEN** the shard executes exactly the `(cell, replicate)` units assigned to index `i` by a deterministic partition of all units
- **AND** the union of units across all `N` shards equals the full set of units with no overlaps

#### Scenario: Shard resumes without duplicating completed work

- **WHEN** a shard is re-run and its JSONL output already contains completed records with matching parameter signatures
- **THEN** the shard skips those replicates and appends only missing or failed ones

#### Scenario: Shard records failures without aborting the study

- **WHEN** a replicate within a shard fails and the configured error policy is to record
- **THEN** the shard writes a failed replicate record with error details and continues with remaining units

### Requirement: Shards merge into a single deduplicated result set

MOTCO SHALL merge per-shard JSONL outputs into a single result set, deduplicating by `(cell_id, replicate_index)` and validating parameter-signature consistency across shards.

#### Scenario: Merge combines all shards

- **WHEN** per-shard JSONL files are merged
- **THEN** the merged result contains exactly one record per `(cell_id, replicate_index)` across all shards

#### Scenario: Merge detects inconsistent shards

- **WHEN** two shards contain the same `(cell_id, replicate_index)` with different parameter signatures
- **THEN** the merge raises a clear validation error rather than silently choosing one record

### Requirement: Study characterizes each statistic independently

MOTCO SHALL characterize the trajectory test using per-statistic operating characteristics, reporting the rejection rate of each of the `delta`, `angle`, and `shape` statistics as its own marginal quantity, without multiplicity correction across statistics.

#### Scenario: Per-statistic rejection rates are reported per cell

- **WHEN** merged results are summarized for a cell
- **THEN** the summary reports, for each of `delta`, `angle`, and `shape`, the completed replicate count, rejection count, rejection rate, and Monte Carlo standard error at the configured alpha

#### Scenario: Unavailable statistic is not counted as a rejection

- **WHEN** a statistic is unavailable for a cell (for example `shape` with fewer than three stages)
- **THEN** the summary reports it as unavailable rather than treating it as non-significant

### Requirement: Study reports a combined-rule Type I result

MOTCO SHALL report, as a secondary result, the Type I error rate of the combined decision rule that rejects when any of the three statistics is significant at the configured alpha, computed only over null cells.

#### Scenario: Combined-rule false-positive rate is computed on null cells

- **WHEN** a null cell (negative control) is summarized under the combined rule
- **THEN** a replicate counts as a rejection if any available statistic's p-value is below alpha
- **AND** the reported rate is the fraction of such replicates with its Monte Carlo standard error

### Requirement: Study produces paper-ready reports

MOTCO SHALL produce, from the merged and summarized results, a mode × statistic specificity matrix, Type I tables, and power-curve data, written as CSV and as figures.

#### Scenario: Specificity matrix is produced

- **WHEN** reporting runs on summarized results
- **THEN** it produces a matrix indexed by trajectory mode and statistic whose entries are rejection rates with Monte Carlo standard errors

#### Scenario: Power curves are produced

- **WHEN** reporting runs on power cells
- **THEN** it produces, for each trajectory mode and statistic, rejection rate as a function of effect size, suitable for plotting as a curve with error bars

#### Scenario: Type I table is produced

- **WHEN** reporting runs on null cells across the configured axes
- **THEN** it produces a table of per-statistic and combined-rule rejection rates with Monte Carlo standard errors

### Requirement: Study evaluates results against pre-specified targets

MOTCO SHALL evaluate summarized results against the acceptance targets declared in the configuration and report, per target, whether it is met given Monte Carlo uncertainty.

#### Scenario: Type I control target is evaluated

- **WHEN** a null cell is evaluated against a Type I control target at alpha
- **THEN** the report indicates whether the empirical rejection rate is within the target's tolerance (expressed in Monte Carlo standard errors) of alpha

#### Scenario: Power monotonicity target is evaluated

- **WHEN** a power mode's diagonal statistic is evaluated against a monotonicity target
- **THEN** the report indicates whether the rejection rate is non-decreasing in effect size and reaches the target's minimum power at the largest effect size

#### Scenario: Specificity target is evaluated

- **WHEN** an off-diagonal mode × statistic combination is evaluated against a specificity target
- **THEN** the report indicates whether its rejection rate stays within tolerance of alpha
