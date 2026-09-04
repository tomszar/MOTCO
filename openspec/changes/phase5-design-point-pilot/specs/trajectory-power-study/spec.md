## MODIFIED Requirements

### Requirement: Study is defined by a declarative configuration

MOTCO SHALL provide a declarative, file-based study configuration that fully determines an enumerated grid of simulation cells, so that a study is reproducible from its config alone.

The configuration MUST capture baseline InterSIM/generator/evaluation parameters, the set of trajectory modes, the set of effect sizes, the one-factor-at-a-time axes with their values, an optional crossed design grid of namespaced axes with their values, the per-cell replicate count, the base seed, and the pre-specified acceptance targets. A declared design grid MUST include the baseline value of every axis it names, and every design-grid axis MUST use the same `generator.` / `evaluation.` namespace rules as the one-factor-at-a-time axes.

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

### Requirement: Reporting resolves orientation operating characteristics by baseline continuity
When a study's enumerated grid varies baseline continuity, study reporting SHALL present the
orientation-relevant operating characteristics as functions of the continuity axis: per-cell
rejection rates for each statistic, the distribution of the recorded pooled configuration eigengap,
and the dispersion of the per-replicate `angle` null width, each resolved by continuity value. The
report MUST make the linking observable explicit — the eigengap is the quantity expected to carry a
continuity-conditioned orientation claim to real data — so a reader can trace power differences
along the axis to the recorded geometry rather than to the knob itself. When the grid also varies
other design coordinates, the continuity-resolved view MUST be resolved on those coordinates as well
and MUST NOT pool records that differ in any design coordinate other than continuity.

#### Scenario: Continuity-resolved orientation table is produced
- **WHEN** a merged result set contains cells that differ only in baseline continuity
- **THEN** the report includes a per-continuity-value summary of orientation rejection rates,
  eigengap distribution summaries, and `angle` null-width dispersion, computed from the persisted
  records alone

#### Scenario: Continuity-resolved table does not pool across other design coordinates
- **WHEN** a merged result set contains design-grid cells that vary baseline continuity and at least one other design coordinate
- **THEN** every row of the continuity-resolved view is identified by continuity together with the values of every other design coordinate
- **AND** no row aggregates records from cells that differ in a design coordinate other than continuity

#### Scenario: Studies without a continuity axis are unaffected
- **WHEN** a study's grid holds baseline continuity fixed
- **THEN** reporting produces its existing outputs unchanged, without a continuity-resolved view

## ADDED Requirements

### Requirement: Crossed design grid enumerates one matched, anchored power grid per design point

When a study configuration declares a design grid, enumeration SHALL cross the declared axes into design points and, for every design point other than the baseline, emit the full power grid — one mode-agnostic zero-effect anchor cell plus one cell per (trajectory mode × nonzero effect size) — with the design point's coordinates applied to the baseline generator and evaluation parameters. The baseline design point MUST be served by the existing primary power cells and shared anchor rather than re-emitted. Every design-point cell MUST carry its design coordinates and its distinct phase in cell metadata so that readers of the primary power curves, the Phase 4 gate, and the acceptance targets do not see design-point cells unless they opt in. Design-point cells MUST share the primary matched-seed family so that cells across design points are paired at the same replicate index, and the existing duplicate-dataset guard and surgery-headroom check MUST apply to every design-point cell using that cell's own parameters.

#### Scenario: Design points cross the declared axes

- **WHEN** a configuration declares a design grid with axes `A` (values a₁…aₖ, including the baseline) and `B` (values b₁…bₘ, including the baseline)
- **THEN** enumeration emits design-point power grids for every (aᵢ, bⱼ) pair except the baseline pair
- **AND** each such grid contains exactly one zero-effect anchor cell and one cell per (mode, nonzero effect)
- **AND** each cell's metadata records the values of both `A` and `B`

#### Scenario: Design-point cells are invisible to baseline readers

- **WHEN** a merged result set from a design-grid study is reported
- **THEN** the primary power curves, specificity matrix, Type I table, Phase 4 gate, and acceptance-target evaluation are computed from the baseline column only and equal what a study without the design grid would report

#### Scenario: Design-point cells share the matched-seed family

- **WHEN** matched seeds are enabled and a design grid is declared
- **THEN** every design-point power cell and anchor draws the same generator seed as the primary cells at the same replicate index
- **AND** enumeration still rejects any two cells in that family whose generator parameters would produce identical datasets

#### Scenario: Headroom is enforced at every design point

- **WHEN** a design point's coordinates (for example a lower baseline continuity) leave a pool-limited surgery's requested effect above the expected headroom at that point
- **THEN** enumeration fails before any compute is spent, naming the offending cell and its saturating effect
- **AND** a design point whose coordinates provide sufficient headroom for the same effect enumerates normally

### Requirement: Reporting resolves operating characteristics by design point

When a merged result set contains design-point cells, study reporting SHALL produce a design-point operating table with one row per (design point, trajectory mode, effect size, statistic), including the baseline column and each design point's zero-effect anchor, carrying the rejection rate with its Monte Carlo standard error, the distribution of the recorded pooled configuration eigengap, the dispersion of the per-replicate `angle` null width, and the distribution of the selected latent dimensionality — all computed from persisted records alone. Studies without design-point cells MUST produce no such table.

#### Scenario: Design-point operating table is produced

- **WHEN** a merged result set from a design-grid study is reported
- **THEN** the report includes a design-point operating table whose rows are identified by every design coordinate, the trajectory mode, the effect size, and the statistic
- **AND** each row carries rejection rate, Monte Carlo standard error, eigengap distribution summaries, `angle` null-width dispersion, and selected-dimensionality summaries

#### Scenario: Design-point nulls are reported

- **WHEN** the design-point operating table is built
- **THEN** every design point contributes rows for the `none` mode at effect `0.0` from its zero-effect anchor, for each statistic

#### Scenario: Studies without a design grid produce no design-point table

- **WHEN** a merged result set contains no design-point cells
- **THEN** reporting produces its existing outputs unchanged and writes no design-point operating table

### Requirement: Design-point decision is predeclared in the configuration and evaluated in the report

A study configuration MAY declare a design-point decision rule naming a target trajectory mode and statistic, a minimum power at the top enumerated effect, a confirmation standard-error threshold, and a preference order over the design-grid axes. When declared, the report SHALL evaluate the rule per design point using the target statistic's rejection rate at the largest effect enumerated in that column, classify each design point as meeting the floor with Monte Carlo confirmation, meeting it marginally, or failing it, and SHALL record a decision: the first confirmed design point in the declared preference order, or a `revise_claim` verdict when none is confirmed. The decision output MUST list every design point's classification and the zero-effect anchor's rejection rate for every statistic at that point, and MUST NOT feed the Phase 4 gate or the acceptance-target report.

#### Scenario: A confirmed design point is chosen in preference order

- **WHEN** at least one design point's target rejection rate minus the confirmation threshold times its Monte Carlo standard error is at or above the floor
- **THEN** the decision names the first such design point in the declared preference order
- **AND** lists every design point's classification and anchor rejection rates

#### Scenario: No design point is confirmed

- **WHEN** no design point meets the floor with confirmation
- **THEN** the decision records the `revise_claim` verdict together with every design point's classification, so the readiness worklist can revise the claim rather than the Monte Carlo size

#### Scenario: Decision thresholds come from the configuration

- **WHEN** the design-point decision is evaluated
- **THEN** the floor, confirmation threshold, target pair, and preference order are read from the configuration's acceptance block
- **AND** no threshold is hard-coded in report code

### Requirement: Study provides a fixed Phase 5 design-point pilot profile

The study SHALL provide a version-controlled Phase 5 design-point pilot configuration using the numpy generator, pooled PLS integration with M-value methylation, four stages, differential-site density `p_dmp = 0.1`, the default fail-loud surgery-censoring policy, a design grid crossing baseline continuity `0.0`, `0.5`, `0.8` with sample sizes `300`, `600`, `1200`, trajectory modes `orientation` and `translation`, effect sizes `0.00`, `0.25`, `0.50`, and `1.00`, 100 replicates per cell, 199 RRPP permutations, matched seeds with a shared zero-effect anchor per design point, attribution disabled, the Phase 4 gate disabled, and a design-point decision rule targeting orientation's `angle` statistic at a 0.80 floor with preference for the smallest sample size and then the lowest continuity. The configuration MUST enumerate without any censored surgery at every design point.

#### Scenario: Phase 5 pilot config is loaded

- **WHEN** the committed Phase 5 design-point pilot configuration is loaded
- **THEN** it deterministically enumerates the baseline Type I controls, the baseline primary power grid, and eight further design-point power grids with the declared modes, effects, replicate count, and permutations
- **AND** every enumerated pool-limited cell fits its expected surgery headroom
- **AND** it requires no R runtime dependency

#### Scenario: Pilot does not copy the historical clamp flag

- **WHEN** the committed Phase 5 design-point pilot configuration is inspected
- **THEN** it does not set `surgery_censoring` to `clamp`

### Requirement: Phase 5 design-point findings are versioned and reproducible

The study SHALL produce a dated design-point findings report tied to the exact configuration, code revision, parameter signatures, software versions, record counts, failure counts, and reproduction commands, recording the design-point decision, the continuity-conditional interpretation with the recorded eigengap as the linking observable, the per-design-point Type I behaviour, and the hand-off of the retained-rank question to the latent-dimensionality readiness item. The Phase 5 readiness worklist MUST record the resulting design point or the revised claim.

#### Scenario: Findings report is committed

- **WHEN** the Phase 5 design-point pilot run and reporting complete
- **THEN** a versioned report records the decision verdict, scientific interpretation, limitations, and exact shard, merge, and report commands
- **AND** all claims in the report can be traced to structured CSV or JSON outputs
- **AND** the readiness worklist's design-point item states the chosen design point or the revised claim
