## ADDED Requirements

### Requirement: Enumeration rejects cells that exceed surgery headroom
Study enumeration SHALL validate, at configuration time and before any compute is spent, that every
cell's requested effect can be realized without censoring in expectation: for each pool-limited
surgery mode, the expected requested surgery size (derived from the cell's generator parameters,
e.g. `p_dmp` and `n_stages`) must not exceed the expected available pool. Enumeration of a config
containing a cell that would saturate SHALL fail with an error naming the cell, its requested
effect, and the effect at which the surgery saturates. Configs that explicitly opt in to the
clamping policy are exempt from the error but MUST still be enumerable.

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

### Requirement: Reports annotate realized effects and duplicate constructions
Study reports SHALL report the realized surgery per power cell (nominal size, mean realized size,
censored fraction) and SHALL flag any pair of power cells whose realized constructions are
identical in distribution — in particular matched-seed cells whose replicates generated identical
datasets — so no two such cells are presented as independent power measurements.

#### Scenario: Realized surgery appears in the study report
- **WHEN** a study report is built from merged records that carry surgery truth metadata
- **THEN** the report includes, per power cell, the nominal surgery size, mean realized surgery
  size, and censored-replicate fraction

#### Scenario: Duplicate realized constructions are flagged
- **WHEN** two power cells in the same seed family share identical realized datasets on more than
  5% of replicate pairs
- **THEN** the report flags the pair as a duplicated construction rather than reporting the two
  cells as independent measurements
