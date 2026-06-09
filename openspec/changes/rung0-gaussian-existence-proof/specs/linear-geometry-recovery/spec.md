## ADDED Requirements

### Requirement: Minimal Gaussian two-stage test bed

The system SHALL provide a generator-free test bed that draws a feature matrix from a multivariate normal distribution with a caller-supplied (or deterministically constructed) covariance, and injects a *known* two-stage trajectory geometry directly in feature space for two groups. The construction MUST be parameterized by an explicit per-group, per-stage feature-space mean configuration so that the intended trajectory geometry is known exactly prior to projection. The test bed MUST NOT invoke the InterSIM generator, the cross-omic cascade, or the methylation `rev.logit` transform.

#### Scenario: Deterministic dataset from a seed

- **WHEN** the test bed is called with a fixed seed, feature dimension, sample sizes, covariance, and a feature-space geometry specification
- **THEN** it returns a feature matrix and group/stage labels whose per-(group, stage) sample means approximate the specified configuration, and repeated calls with the same seed return identical matrices

#### Scenario: Exactly two stages

- **WHEN** a dataset is generated
- **THEN** there are exactly two stages per group, so each group's trajectory is a single step vector and Procrustes shape is degenerate and excluded from all reported statistics

### Requirement: Linear (PCA) projection and geometry measurement

The system SHALL project the test-bed feature matrix into a latent space with PCA (a pure linear projector) fit inline, and SHALL measure the between-group magnitude (`delta`) and orientation (`angle`) of the two-stage trajectories using the existing `stats/trajectory.py` estimators. The measurement MUST operate on the PCA-projected latent coordinates and MUST report `delta` and `angle` only (no shape).

#### Scenario: Magnitude and orientation are reported from the latent space

- **WHEN** a generated dataset is projected with PCA and evaluated
- **THEN** the result includes a `delta` value and an `angle` value computed from the projected two-stage trajectories, and no shape value is produced

### Requirement: Clean-floor existence proof for magnitude and orientation

The system SHALL demonstrate that, under the linear PCA projector, a pure-magnitude feature-space manipulation (group B's step is a scalar multiple of group A's step) registers as a change in `delta` while leaving `angle` near its null, and a pure-orientation feature-space manipulation (group B's step is a rotation of group A's, preserving length) registers as a change in `angle` while leaving `delta` near its null. Cross-talk MUST be near zero in both directions, establishing the clean floor.

#### Scenario: Pure-magnitude manipulation moves delta only

- **WHEN** group B's feature-space step is constructed as `c·a` for a scalar `c ≠ 1` (same direction, scaled length) and the dataset is projected and evaluated
- **THEN** the measured `delta` reflects the intended magnitude difference and the measured `angle` remains near zero (within a small tolerance)

#### Scenario: Pure-orientation manipulation moves angle only

- **WHEN** group B's feature-space step is constructed as `R·a` for a rotation `R` that preserves length and the dataset is projected and evaluated
- **THEN** the measured `angle` reflects the intended orientation difference and the measured `delta` remains near zero (within a small tolerance)

### Requirement: Exact inverse design of feature-space changes

The system SHALL compute, given group A's latent step vector `a` and the fitted PCA loadings `Vₖ`, the minimum-norm feature-space change `Δx` that produces a specified target latent step: for a pure-magnitude target `b = c·a`, `Δx = Vₖ·(c−1)·a`; for a pure-orientation target `b = R·a`, `Δx = Vₖ·(R−I)·a`. The system SHALL expose the resulting `Δx` for inspection, including its support and a sparsity/concentration summary, so the feature-space interpretation of each geometric facet can be examined.

#### Scenario: Magnitude target yields a same-direction feature change

- **WHEN** a pure-magnitude latent target `b = c·a` is requested
- **THEN** the returned `Δx` projects (via the PCA map) to `(c−1)·a` within numerical tolerance, and its feature-space readout can be compared against group A's step direction

#### Scenario: Orientation target yields a feature-mixing rotation

- **WHEN** a pure-orientation latent target `b = R·a` is requested
- **THEN** the returned `Δx` projects to `(R−I)·a` within numerical tolerance, and its support/sparsity summary is reported so it can be judged whether the change concentrates on a disjoint feature set or mixes features within the loading subspace

### Requirement: Findings writeup

The system SHALL produce a committed findings writeup that records the intended-geometry→measured outcome (a table covering the pure-magnitude and pure-orientation manipulations with their measured `delta`/`angle`) and the inverse-design feature recipes (the `Δx` support/sparsity summaries for magnitude and orientation), with enough parameter detail to reproduce the run.

#### Scenario: Findings document exists and is reproducible

- **WHEN** the test bed and inverse-design analysis have been run
- **THEN** a findings document records the measured table, the inverse-design recipes, and the exact parameters (seed, dimensions, covariance, number of PCA components) needed to reproduce them
