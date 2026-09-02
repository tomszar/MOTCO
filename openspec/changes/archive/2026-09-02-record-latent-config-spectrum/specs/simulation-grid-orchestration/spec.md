## ADDED Requirements

### Requirement: Replicate records persist the configuration spectrum

Persisted replicate records SHALL carry the evaluation's configuration-spectrum block — pooled and per-group observed spectra and, when permutations were run, the permutation eigengap summary — beside the existing null summary. Records written before the field existed MUST still load, with the block empty. The cell parameter signature MUST incorporate an explicit spectrum schema version so that resume against records written under a different spectrum contract is refused rather than producing a result set in which only some records carry the field.

#### Scenario: Completed replicate persists its spectrum

- **WHEN** a replicate completes through the grid runner
- **THEN** its persisted record carries the configuration-spectrum block from the evaluation result
- **AND** the block round-trips through serialization unchanged

#### Scenario: Legacy record loads without the field

- **WHEN** a reader loads a record written before the spectrum field existed
- **THEN** the record loads with an empty spectrum block
- **AND** it is distinguishable from a record whose evaluation produced a degenerate spectrum

#### Scenario: Pre-spectrum shards refuse resume

- **WHEN** a run attempts to resume against records whose parameter signature lacks the spectrum schema version
- **THEN** resume is refused with the existing signature-mismatch behavior rather than mixing contracts

#### Scenario: Signature is stable for the new contract

- **WHEN** the same cell is enumerated twice under the new schema
- **THEN** its parameter signature is identical, and resume against records written under the new schema succeeds
