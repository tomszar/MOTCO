## MODIFIED Requirements

### Requirement: Harness builds MOTCO trajectory design objects
The harness SHALL construct model matrices, LS means, and trajectory contrasts from generated sample metadata. Level ordering SHALL be numeric-aware: when every distinct label of a factor (group or stage) parses as an integer, the levels are ordered by numeric value; otherwise the levels keep the deterministic lexicographic order of their string representations. The same ordering rule MUST govern the model matrix columns, the LS-mean rows, and the contrast indices so they stay mutually consistent.

#### Scenario: Design objects are derived from metadata
- **WHEN** the dataset metadata contains valid `group` and `stage` columns
- **THEN** the harness builds full and reduced model matrices, LS means, and a two-group trajectory contrast

#### Scenario: Missing metadata columns are rejected
- **WHEN** required metadata columns are missing
- **THEN** the harness raises a clear validation error

#### Scenario: Ten or more integer-labeled stages order numerically
- **WHEN** the metadata's stage labels are the strings "0" through "11"
- **THEN** the design's stage-level order is 0, 1, 2, ..., 11 (numeric), not the lexicographic order that places "10" and "11" before "2", and LS-mean rows and contrast indices follow the same order

#### Scenario: Non-numeric labels keep lexicographic order
- **WHEN** any factor level label does not parse as an integer
- **THEN** that factor's levels are ordered lexicographically by string representation, identical to the previous behavior
