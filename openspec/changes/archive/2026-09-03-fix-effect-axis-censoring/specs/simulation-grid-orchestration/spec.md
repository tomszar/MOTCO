## ADDED Requirements

### Requirement: Cell summaries surface realized surgery
Cell-level summaries SHALL surface the realized surgery recorded in replicate truth metadata beside
rejection metrics: per cell, the nominal surgery size, the mean realized surgery size, and the
fraction of replicates whose surgery was censored. Cells whose generator mode has no pool-limited
surgery (e.g. `magnitude`, `none`) report the realized-surgery fields as absent rather than zero.

#### Scenario: Summary reports nominal and realized surgery per cell
- **WHEN** replicate records carrying pool-limited surgery truth are summarized
- **THEN** each cell's summary includes the nominal surgery size, the mean realized surgery size,
  and the censored-replicate fraction alongside the rejection metrics

#### Scenario: Modes without pool-limited surgery are not misreported
- **WHEN** replicate records from a mode with no pool-limited surgery are summarized
- **THEN** the realized-surgery fields are reported as unavailable, not as zero-size surgeries
