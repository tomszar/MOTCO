## RENAMED Requirements

- FROM: `### Requirement: Shape removes location, proper rotation, and uniform scale`
- TO: `### Requirement: Shape removes location, orthogonal transformation, and uniform scale`

- FROM: `### Requirement: Reflection policy is explicit`
- TO: `### Requirement: Reflection policy is uniform across ambient dimensions`

## MODIFIED Requirements

### Requirement: Shape removes location, orthogonal transformation, and uniform scale

MOTCO SHALL report zero trajectory `shape` distance, within documented numerical tolerance, for any two trajectories whose stage configurations differ only by translation, orthogonal transformation (rotation or reflection), and uniform scale.

#### Scenario: Translated trajectory has zero shape distance

- **WHEN** two trajectories have the same stage configuration and one trajectory is translated by a constant vector
- **THEN** their pairwise `shape` distance is zero within numerical tolerance

#### Scenario: Uniformly scaled trajectory has zero shape distance

- **WHEN** two trajectories have the same stage configuration and one trajectory is uniformly scaled by a positive scalar
- **THEN** their pairwise `shape` distance is zero within numerical tolerance

#### Scenario: Rigidly rotated trajectory has zero shape distance

- **WHEN** two trajectories have the same stage configuration and one trajectory is transformed by a proper rigid rotation
- **THEN** their pairwise `shape` distance is zero within numerical tolerance

#### Scenario: Reflected trajectory has zero shape distance

- **WHEN** two trajectories have the same stage configuration and one trajectory is a mirror reflection of the other
- **THEN** their pairwise `shape` distance is zero within numerical tolerance

### Requirement: Shape detects residual configuration changes

MOTCO SHALL report a positive trajectory `shape` distance when trajectories retain a residual configuration difference after translation, orthogonal transformation, and uniform scale have been removed.

#### Scenario: Middle-stage bend has positive shape distance

- **WHEN** a trajectory with at least three stages has one interior stage displaced so that the result cannot be recovered from the original by translation, orthogonal transformation, and uniform scale
- **THEN** its pairwise `shape` distance from the original trajectory is positive

#### Scenario: Stage distance ratios differ

- **WHEN** two trajectories have different relative stage-to-stage distance ratios after normalizing total size
- **THEN** their pairwise `shape` distance is positive

### Requirement: Reflection policy is uniform across ambient dimensions

MOTCO SHALL align reflections away in the trajectory `shape` statistic at every ambient dimension: the Procrustes alignment optimizes over the full orthogonal group, so the reported shape distance MUST NOT depend on whether the stage configuration spans the ambient space. The single policy MUST hold identically at every measurement checkpoint (population, standardized-observed, and latent spaces), and it MUST be documented and tested in both the full-rank and rank-deficient regimes.

#### Scenario: Reflected trajectory follows documented policy

- **WHEN** one trajectory is a mirror reflection of another trajectory
- **THEN** the reported `shape` distance follows the documented reflection policy: the reflection is aligned away and the distance is zero within numerical tolerance

#### Scenario: Mirror pair at full rank has zero shape distance

- **WHEN** one trajectory is a mirror reflection of another and the stage configuration spans the ambient space (configuration rank equals ambient dimension)
- **THEN** the reported `shape` distance is zero within numerical tolerance

#### Scenario: Mirror pair in a higher-dimensional embedding has zero shape distance

- **WHEN** one trajectory is a mirror reflection of another and both are embedded in an ambient space of dimension strictly greater than the configuration rank
- **THEN** the reported `shape` distance is zero within numerical tolerance

#### Scenario: Shape distance is invariant to embedding dimension

- **WHEN** the same pair of trajectory configurations is evaluated at its native dimension and again after zero-padding into a higher-dimensional ambient space
- **THEN** the two reported `shape` distances are equal within numerical tolerance

### Requirement: Shape remains separate from magnitude and orientation statistics

MOTCO SHALL preserve the semantic separation between trajectory magnitude, orientation, and shape so that pure magnitude or pure orthogonal-transformation differences do not produce nonzero `shape` distance.

#### Scenario: Magnitude-only contrast does not produce shape

- **WHEN** two trajectories differ only by uniform trajectory scale
- **THEN** `delta` may be nonzero and `shape` is zero within numerical tolerance

#### Scenario: Orientation-only contrast does not produce shape

- **WHEN** two trajectories differ only by an orthogonal transformation
- **THEN** `angle` may be nonzero and `shape` is zero within numerical tolerance
