## ADDED Requirements

### Requirement: Replicate records persist the permutation null summary

Each persisted replicate record SHALL carry the evaluation's per-statistic permutation null summary so the relationship between a replicate's observed statistic and its own null survives into the study records. The field MUST be additive: records written before it existed MUST still load, and a record from an evaluation that ran no permutations MUST be distinguishable from one whose summary is missing.

The null summary MUST NOT enter the cell parameter signature, because it is derived from permutation draws that already occurred and changes no generation, integration, or permutation behavior. A run resumed against records written before this field MUST therefore remain resumable without an overwrite.

#### Scenario: Summary reaches the persisted record

- **WHEN** a replicate completes with permutation count greater than 0
- **THEN** its persisted record carries the per-statistic null summary from the evaluation

#### Scenario: Legacy records still load

- **WHEN** replicate records written before the null summary field existed are read back
- **THEN** they load successfully with an empty null summary

#### Scenario: Resume is unaffected

- **WHEN** a run resumes against existing records whose cells are otherwise unchanged
- **THEN** the parameter signature still matches and completed replicates are skipped without overwrite
