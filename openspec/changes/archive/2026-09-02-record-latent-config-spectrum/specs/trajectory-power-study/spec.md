## ADDED Requirements

### Requirement: Reporting stratifies orientation power by configuration eigengap

The study report SHALL summarize the recorded pooled and per-group eigengaps per cell beside the existing geometry summaries, and for orientation-mode power cells SHALL report rejection rates stratified by the recorded pooled eigengap, with per-stratum replicate counts and Monte Carlo uncertainty. Stratification MUST read recorded values only — the report MUST NOT regenerate datasets or recompute spectra from raw data.

#### Scenario: Eigengap summaries join the cell tables

- **WHEN** merged records carrying the spectrum block are reported
- **THEN** per-cell tables include summaries of the recorded pooled and per-group eigengaps

#### Scenario: Orientation power is stratified

- **WHEN** an orientation-mode power cell with recorded spectra is reported
- **THEN** the report includes rejection rates by eigengap stratum with per-stratum counts and uncertainty
- **AND** the strata are defined from the recorded values, not from regenerated data

#### Scenario: Legacy records degrade gracefully

- **WHEN** a report runs over merged records lacking the spectrum block
- **THEN** the stratified table is reported as unavailable for those cells and every pre-existing table is produced unchanged
