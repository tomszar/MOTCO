## ADDED Requirements

### Requirement: Replicate seeds are R-compatible

The orchestrator's replicate-seed derivation SHALL produce values in
the range `[0, 2³¹ − 1]` so that the same seed can be consumed by R
(via `set.seed`), Python `numpy`, and any other downstream RNG
without coercion or overflow.

#### Scenario: Derived seed is within R's signed-32-bit range

- **WHEN** the orchestrator derives a replicate seed for any
  `(cell, replicate_index)` pair
- **THEN** the returned seed is an integer in `[0, 2³¹ − 1]`

#### Scenario: Seed derivation is deterministic

- **WHEN** the same `(base_seed, cell_id, replicate_index)` triple is
  used to derive a replicate seed in different processes
- **THEN** the returned seed is identical

### Requirement: Parameter signature invalidates legacy seed derivations

The orchestrator's parameter signature SHALL include a
seed-derivation version tag so that completed replicate records
produced under a previous seed-derivation function are detected as
mismatched on resume.

#### Scenario: Signature changes when seed derivation changes

- **WHEN** the seed-derivation function is updated and the
  derivation-version tag is bumped
- **THEN** previously completed shards have a different parameter
  signature for the same cell and the resume guard refuses to skip
  those replicates

#### Scenario: Signature is stable across processes for the same derivation version

- **WHEN** the same cell is signed in different processes using the
  same derivation-version tag
- **THEN** the produced signatures are identical
