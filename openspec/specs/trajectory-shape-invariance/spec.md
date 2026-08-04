# trajectory-shape-invariance Specification

## Purpose
Defines the production contract for MOTCO trajectory shape as a strict geometric-morphometric quantity: the residual trajectory configuration after removing location, proper rigid orientation, and uniform size.
## Requirements
### Requirement: Shape removes location, proper rotation, and uniform scale

MOTCO SHALL report zero trajectory `shape` distance, within documented numerical tolerance, for any two trajectories whose stage configurations differ only by translation, proper rigid rotation, and uniform scale.

#### Scenario: Translated trajectory has zero shape distance

- **WHEN** two trajectories have the same stage configuration and one trajectory is translated by a constant vector
- **THEN** their pairwise `shape` distance is zero within numerical tolerance

#### Scenario: Uniformly scaled trajectory has zero shape distance

- **WHEN** two trajectories have the same stage configuration and one trajectory is uniformly scaled by a positive scalar
- **THEN** their pairwise `shape` distance is zero within numerical tolerance

#### Scenario: Rigidly rotated trajectory has zero shape distance

- **WHEN** two trajectories have the same stage configuration and one trajectory is transformed by a proper rigid rotation
- **THEN** their pairwise `shape` distance is zero within numerical tolerance

### Requirement: Shape detects residual configuration changes

MOTCO SHALL report a positive trajectory `shape` distance when trajectories retain a residual configuration difference after translation, proper rotation, and uniform scale have been removed.

#### Scenario: Middle-stage bend has positive shape distance

- **WHEN** a trajectory with at least three stages has one interior stage displaced so that the result cannot be recovered from the original by translation, proper rotation, and uniform scale
- **THEN** its pairwise `shape` distance from the original trajectory is positive

#### Scenario: Stage distance ratios differ

- **WHEN** two trajectories have different relative stage-to-stage distance ratios after normalizing total size
- **THEN** their pairwise `shape` distance is positive

### Requirement: Reflection policy is explicit

MOTCO SHALL define whether mirror-reflected trajectory configurations are aligned away or retained as distinct shapes, and the default production behavior MUST be documented and tested.

#### Scenario: Reflected trajectory follows documented policy

- **WHEN** one trajectory is a mirror reflection of another trajectory
- **THEN** the reported `shape` distance follows the documented reflection policy

### Requirement: Shape remains separate from magnitude and orientation statistics

MOTCO SHALL preserve the semantic separation between trajectory magnitude, orientation, and shape so that pure magnitude or pure proper-rotation differences do not produce nonzero `shape` distance.

#### Scenario: Magnitude-only contrast does not produce shape

- **WHEN** two trajectories differ only by uniform trajectory scale
- **THEN** `delta` may be nonzero and `shape` is zero within numerical tolerance

#### Scenario: Orientation-only contrast does not produce shape

- **WHEN** two trajectories differ only by a proper rigid rotation
- **THEN** `angle` may be nonzero and `shape` is zero within numerical tolerance

### Requirement: Shape audit produces reproducible diagnostic evidence

MOTCO SHALL include deterministic diagnostic cases that compare observed `shape` behavior with the strict morphometric invariance contract and record any divergence from legacy GPA behavior.

#### Scenario: Audit records invariant cases

- **WHEN** the shape invariance audit is run
- **THEN** it records outcomes for translation, uniform scale, proper rotation, reflection, and genuine bend cases

#### Scenario: Legacy divergence is visible

- **WHEN** legacy GPA behavior differs from strict morphometric shape behavior
- **THEN** the audit output identifies the affected transformation case and the observed distance under each method

