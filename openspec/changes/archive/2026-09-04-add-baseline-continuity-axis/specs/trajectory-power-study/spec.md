## MODIFIED Requirements

### Requirement: Enumeration rejects cells that exceed surgery headroom
Study enumeration SHALL validate, at configuration time and before any compute is spent, that every
cell's requested effect can be realized without censoring in expectation: for each pool-limited
surgery mode, the expected requested surgery size (derived from the cell's generator parameters,
e.g. `p_dmp`, `n_stages`, and baseline continuity) must not exceed the expected available pool,
with the expectation computed under the cell's own baseline continuity value (the
continuity-adjusted stage-active union, reducing to the independence union at zero continuity).
Enumeration of a config containing a cell that would saturate SHALL fail with an error naming the
cell, its requested effect, and the effect at which the surgery saturates. Configs that explicitly
opt in to the clamping policy are exempt from the error but MUST still be enumerable.

#### Scenario: Censored cell fails enumeration
- **WHEN** a study config requests an orientation or translation effect beyond the expected
  headroom of its generator parameters under the default (fail-loud) censoring policy
- **THEN** enumeration raises a configuration error naming the offending cell, the requested
  effect, and the saturating effect, before any replicate is run

#### Scenario: Phase-4 pilot grid is caught
- **WHEN** the Phase-4 pilot configuration (orientation and translation at effects 0.75 and 1.00
  with `p_dmp = 0.2`, four stages) is enumerated under the default policy
- **THEN** enumeration fails on the censored cells rather than producing a grid whose top cells
  regenerate near-identical datasets

#### Scenario: Headroom-respecting grids enumerate unchanged
- **WHEN** every cell's requested effect is within the expected surgery headroom
- **THEN** enumeration succeeds and produces the same cells, seeds, and signatures it would have
  apart from the generator's new policy parameter

#### Scenario: Headroom is evaluated per continuity value
- **WHEN** a study config sweeps baseline continuity as a generator axis alongside a pool-limited
  surgery mode
- **THEN** each enumerated cell's headroom check uses that cell's own continuity value, so an
  effect rejected at zero continuity may enumerate at a higher continuity where the expected pool
  is larger

## ADDED Requirements

### Requirement: Reporting resolves orientation operating characteristics by baseline continuity
When a study's enumerated grid varies baseline continuity, study reporting SHALL present the
orientation-relevant operating characteristics as functions of the continuity axis: per-cell
rejection rates for each statistic, the distribution of the recorded pooled configuration eigengap,
and the dispersion of the per-replicate `angle` null width, each resolved by continuity value. The
report MUST make the linking observable explicit — the eigengap is the quantity expected to carry a
continuity-conditioned orientation claim to real data — so a reader can trace power differences
along the axis to the recorded geometry rather than to the knob itself.

#### Scenario: Continuity-resolved orientation table is produced
- **WHEN** a merged result set contains cells that differ only in baseline continuity
- **THEN** the report includes a per-continuity-value summary of orientation rejection rates,
  eigengap distribution summaries, and `angle` null-width dispersion, computed from the persisted
  records alone

#### Scenario: Studies without a continuity axis are unaffected
- **WHEN** a study's grid holds baseline continuity fixed
- **THEN** reporting produces its existing outputs unchanged, without a continuity-resolved view
