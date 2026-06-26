## ADDED Requirements

### Requirement: Integration converts methylation to M-value space
The production pipeline SHALL convert `SemiSyntheticTrajectoryDataset.methylation` (B values) to M-value (logit) space before any integration processing, using a clipped logit transform with clip=1e-6.

#### Scenario: M-value conversion applied before concatenation
- **WHEN** the integration method is `concat`
- **THEN** the methylation block is logit-transformed before standardisation and column binding

#### Scenario: M-value conversion applied before SNF
- **WHEN** the integration method is `snf`
- **THEN** the methylation layer is logit-transformed before affinity computation

#### Scenario: M-value conversion applied before PLS
- **WHEN** the integration method is `pls`
- **THEN** the methylation block is logit-transformed before standardisation and PLS-DA fitting

#### Scenario: Expression and proteomics layers are not transformed
- **WHEN** any integration method processes the dataset
- **THEN** the expression and proteomics layers are used as-is without logit transformation

### Requirement: Canonical logit lives in the generator module
The `logit(x, clip)` function SHALL be defined in `generator.py` as the exact inverse of `rev_logit`, and SHALL be the single source of truth used by all consumers.

#### Scenario: logit is importable from generator
- **WHEN** a module needs the clipped logit transform
- **THEN** it imports `logit` from `motco.simulations.generator`

#### Scenario: methylation_recovery uses the canonical logit
- **WHEN** `methylation_recovery.beta_to_mvalue` is called
- **THEN** it delegates to `generator.logit` (no duplicated implementation)