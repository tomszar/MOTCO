## MODIFIED Requirements

### Requirement: Study is defined by a declarative configuration

MOTCO SHALL provide a declarative, file-based study configuration that fully determines an enumerated grid of simulation cells, so that a study is reproducible from its config alone.

The configuration MUST capture baseline InterSIM/generator/evaluation parameters, the set of trajectory modes, the set of effect sizes, the one-factor-at-a-time axes with their values, an optional crossed design grid of namespaced axes with their values, the per-cell replicate count, the base seed, and the pre-specified acceptance targets. A declared design grid MUST include the baseline value of every axis it names, and every design-grid axis MUST use the same `generator.` / `evaluation.` namespace rules as the one-factor-at-a-time axes. Both kinds of axis MAY name a nested evaluation integration parameter (`evaluation.integration_params.<key>`); for such an axis the baseline value is the mapping's current value or `null` when the key is absent, `null` is a legal axis value meaning "key absent", and the baseline-value rule applies to that resolved value.

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

#### Scenario: Design grid without its baseline column is rejected

- **WHEN** a study configuration declares a design grid whose values for some axis omit that axis's baseline value, or names an axis with an unknown namespace, or names an axis also declared as a one-factor-at-a-time axis
- **THEN** loading the configuration raises a clear validation error identifying the offending axis

#### Scenario: Configuration without a design grid is unchanged

- **WHEN** a study configuration declares no design grid
- **THEN** enumeration emits exactly the cells it emitted before the design grid existed, with identical identifiers and parameter signatures

#### Scenario: Nested evaluation axis resolves its baseline from the integration parameters

- **WHEN** a study configuration declares the design-grid axis `evaluation.integration_params.forced_components` with values `[null, 4, 6]` and its baseline evaluation integration parameters do not set that key
- **THEN** the configuration loads, the axis's baseline value is `null`, and the `null` value survives a configuration dump and reload unchanged

#### Scenario: Nested evaluation axis without its baseline value is rejected

- **WHEN** a study configuration declares `evaluation.integration_params.forced_components` with values `[4, 6]` while the baseline does not set the key
- **THEN** loading the configuration raises a clear validation error naming the axis and the missing baseline value `null`

### Requirement: Reporting resolves operating characteristics by design point

When a merged result set contains design-point cells, study reporting SHALL produce a design-point operating table with one row per (design point, trajectory mode, effect size, statistic), including the baseline column and each design point's zero-effect anchor, carrying the rejection rate with its Monte Carlo standard error, the distribution of the recorded pooled configuration eigengap, the dispersion of the per-replicate `angle` null width, the distribution of the selected latent dimensionality, and the component-selection mode (cross-validated or forced) of the records in that row — all computed from persisted records alone. Studies without design-point cells MUST produce no such table.

#### Scenario: Design-point operating table is produced

- **WHEN** a merged result set from a design-grid study is reported
- **THEN** the report includes a design-point operating table whose rows are identified by every design coordinate, the trajectory mode, the effect size, and the statistic
- **AND** each row carries rejection rate, Monte Carlo standard error, eigengap distribution summaries, `angle` null-width dispersion, selected-dimensionality summaries, and the component-selection mode

#### Scenario: Design-point nulls are reported

- **WHEN** the design-point operating table is built
- **THEN** every design point contributes rows for the `none` mode at effect `0.0` from its zero-effect anchor, for each statistic

#### Scenario: Forced-rank rows are marked as such

- **WHEN** a design column's records were evaluated with a forced retained rank
- **THEN** every row of that column reports the component-selection mode as forced and its selected-dimensionality summary equals the forced rank
- **AND** rows from the cross-validated baseline column report the mode as cross-validated

#### Scenario: Studies without a design grid produce no design-point table

- **WHEN** a merged result set contains no design-point cells
- **THEN** reporting produces its existing outputs unchanged and writes no design-point operating table

## ADDED Requirements

### Requirement: Evaluation-only design columns share datasets by design

When two design points differ only in evaluation-namespace coordinates, enumeration SHALL place them in the same matched-seed family with identical generator parameters, so that at every replicate index they evaluate the same generated dataset under different measurement settings, and the duplicate-dataset guard MUST accept them. The report MUST state, wherever such columns are compared, that the comparison is paired on identical datasets and differs only in measurement.

#### Scenario: Rank columns evaluate identical datasets

- **WHEN** a design grid declares only the axis `evaluation.integration_params.forced_components`
- **THEN** every design column's power cells and anchor share the generator parameters and replicate seeds of the baseline column's corresponding cells
- **AND** enumeration succeeds without a duplicate-dataset error

#### Scenario: Generator-identical columns that also share evaluation settings are still rejected

- **WHEN** two cells in one matched-seed family have identical generator parameters and identical evaluation parameters
- **THEN** enumeration fails with the existing duplicate-dataset error

### Requirement: Report renders the retained-rank ladder

When a study's design grid declares the retained-rank axis, the report SHALL render a rank-ladder figure showing, per trajectory mode at the largest enumerated effect and for the zero-effect anchor, each statistic's rejection rate against the retained rank with Monte Carlo error bars, with the cross-validated baseline column placed at its recorded median selected rank and marked as cross-validated. The figure MUST be built from the design-point operating table alone, and studies without the rank axis MUST produce no such figure.

#### Scenario: Rank-ladder figure is produced

- **WHEN** a merged result set from a study declaring `evaluation.integration_params.forced_components` as a design axis is reported
- **THEN** a rank-ladder figure is written beside the design-point operating table
- **AND** the cross-validated column is visibly distinguished from the forced-rank columns

#### Scenario: Studies without the rank axis produce no ladder figure

- **WHEN** a merged result set from a design-grid study that does not declare the rank axis is reported
- **THEN** no rank-ladder figure is written and every existing output is unchanged

### Requirement: Retained-rank decision is predeclared in the configuration and evaluated against the baseline column

A study configuration MAY declare a retained-rank decision rule naming the rank axis, a target (trajectory mode, statistic) pair, one or more protected (trajectory mode, statistic) pairs, a Type I bound for the zero-effect anchor, and separate standard-error multipliers for the target gain and the protected loss. When declared, the report SHALL evaluate the rule with the cross-validated column (rank value `null`) as the reference: a forced-rank column *qualifies* when (a) its target rejection rate at the largest enumerated effect exceeds the reference rate by more than the gain multiplier times the pooled Monte Carlo standard error of the difference, (b) its zero-effect anchor's rejection rate is within the Type I bound for every statistic, and (c) none of its protected rejection rates at the largest enumerated effect falls below the reference rate by more than the loss multiplier times the pooled standard error of the difference. The decision MUST be `keep_cv` when no column qualifies and `adopt_fixed_rank` naming the smallest qualifying rank otherwise. The output MUST list every column's per-criterion status and the quantities each criterion was computed from, MUST read every threshold from the configuration, and MUST NOT feed the Phase 4 gate or the acceptance-target report.

#### Scenario: No rank qualifies

- **WHEN** every forced-rank column fails at least one of the three criteria
- **THEN** the decision records `keep_cv` together with every column's per-criterion status

#### Scenario: Smallest qualifying rank is adopted

- **WHEN** ranks 6 and 9 both satisfy all three criteria and rank 4 does not
- **THEN** the decision records `adopt_fixed_rank` with rank 6

#### Scenario: A protected statistic vetoes an otherwise better rank

- **WHEN** a forced-rank column clears the target gain and the Type I bound but lowers a protected rejection rate below the reference by more than the loss multiplier times the standard error
- **THEN** that column does not qualify and the protected pair is named as the failing criterion

#### Scenario: Anchor inflation vetoes a rank

- **WHEN** a forced-rank column clears the target gain but its zero-effect anchor exceeds the Type I bound on any statistic
- **THEN** that column does not qualify and the inflated statistic is named

#### Scenario: Decision thresholds come from the configuration

- **WHEN** the retained-rank decision is evaluated
- **THEN** the axis, target pair, protected pairs, Type I bound, and both multipliers are read from the configuration's acceptance block
- **AND** declaring the rule without the named rank axis in the design grid is rejected at configuration load

### Requirement: Study provides a fixed Phase 5 latent-rank ladder profile

The study SHALL provide a version-controlled Phase 5 latent-rank ladder configuration using the numpy generator, pooled PLS integration with M-value methylation, four stages, `n_samples = 1200`, baseline continuity `0.0`, differential-site density `p_dmp = 0.1`, the default fail-loud surgery-censoring policy, trajectory modes `magnitude`, `orientation`, `shape`, and `translation`, effect sizes `0.00`, `0.25`, `0.50`, and `1.00`, a design grid over `evaluation.integration_params.forced_components` with values `null`, `3`, `4`, `6`, `9`, and `12`, 100 replicates per cell, 199 RRPP permutations, matched seeds with a shared zero-effect anchor per design point, attribution disabled, the Phase 4 gate disabled, and a retained-rank decision rule targeting orientation's `angle` statistic while protecting magnitude's `delta` and shape's `shape`. The configuration MUST enumerate without any censored surgery.

#### Scenario: Ladder config is loaded

- **WHEN** the committed Phase 5 latent-rank ladder configuration is loaded
- **THEN** it deterministically enumerates the baseline Type I controls, the baseline primary power grid, and five further design-point power grids with the declared modes, effects, replicate count, and permutations
- **AND** every enumerated pool-limited cell fits its expected surgery headroom
- **AND** every design column shares generator parameters and seeds with the baseline column
- **AND** it requires no R runtime dependency

#### Scenario: Ladder does not copy the historical clamp flag

- **WHEN** the committed Phase 5 latent-rank ladder configuration is inspected
- **THEN** it does not set `surgery_censoring` to `clamp`

### Requirement: Phase 5 latent-rank findings are versioned and the readiness worklist records the committed rank rule

The study SHALL produce a dated latent-rank findings report tied to the exact configuration, code revision, parameter signatures, software versions, record counts, failure counts, and reproduction commands, recording the retained-rank decision, each statistic's response to rank per mode beside the recorded eigengap and `angle` null-width dispersion, whether the orientation→shape response decays with rank at the chosen design point, the re-measured magnitude and shape operating characteristics at the chosen design point under the cross-validated rank, and the group-blind statement of the committed rank rule. The Phase 5 readiness worklist MUST record the committed rank rule as the integration configuration Phase 5 runs with, and the roadmap's Phase 5 planned baseline MUST name it.

#### Scenario: Findings report is committed

- **WHEN** the Phase 5 latent-rank ladder run and reporting complete
- **THEN** a versioned report records the decision verdict, scientific interpretation, limitations, and exact shard, merge, and report commands
- **AND** all claims in the report can be traced to structured CSV or JSON outputs
- **AND** the readiness worklist's latent-dimensionality item states the committed rank rule and the roadmap's Phase 5 planned baseline names it
