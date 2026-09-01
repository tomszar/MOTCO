## Purpose

Defines the production contract for MOTCO trajectory `angle` as a difference in direction of progression: the estimator that summarizes a multi-stage trajectory's direction, the transformations that direction must ignore, and the null configurations that must never be reported as large orientation differences.

## ADDED Requirements

### Requirement: Orientation is the principal axis of the stage configuration, signed by progression

MOTCO SHALL estimate a trajectory's orientation as the leading principal axis of its centered stage configuration, with the sign resolved so the returned vector points along the trajectory's direction of progression — from its first stage toward its last. The estimator MUST return a direction, not an axis: reversing the stage order MUST reverse the returned vector.

#### Scenario: Orientation follows the direction of progression

- **WHEN** a trajectory's orientation is estimated
- **THEN** the returned vector has a non-negative inner product with the trajectory's net displacement from its first stage to its last

#### Scenario: Reversing stage order reverses orientation

- **WHEN** the same stage configuration is presented in reversed stage order
- **THEN** the returned orientation vector is negated

#### Scenario: Two-stage orientation is the transition direction

- **WHEN** a trajectory has exactly two stages
- **THEN** its orientation is the unit vector of the transition from the first stage to the second

### Requirement: Orientation is invariant to translation and uniform scale

MOTCO SHALL report a trajectory `angle` of zero, within documented numerical tolerance, between any two trajectories whose stage configurations differ only by translation, by uniform scale, or by both. The angle MUST NOT depend on where the trajectories sit relative to the coordinate origin.

#### Scenario: Translated trajectory has zero angle

- **WHEN** two trajectories have the same stage configuration and one is translated by a constant vector
- **THEN** their pairwise `angle` is zero within numerical tolerance

#### Scenario: Trajectories on opposite sides of the origin have zero angle

- **WHEN** two trajectories with the same stage configuration are translated so they lie on opposite sides of the coordinate origin
- **THEN** their pairwise `angle` is zero within numerical tolerance

#### Scenario: Uniformly scaled trajectory has zero angle

- **WHEN** two trajectories have the same stage configuration and one is uniformly scaled by a positive factor
- **THEN** their pairwise `angle` is zero within numerical tolerance

### Requirement: Bent trajectories do not produce spurious reversals

The orientation sign MUST remain stable for trajectories whose stage configuration bends — including configurations that depart from and return toward their starting region, where the first stage lies near the configuration centroid along the principal axis. Two trajectories that carry no orientation difference MUST NOT be reported as near-antiparallel.

#### Scenario: Bent identical trajectories report zero angle

- **WHEN** two copies of a bent multi-stage trajectory whose first stage projects near zero onto its own centered principal axis are compared
- **THEN** their pairwise `angle` is zero within numerical tolerance

#### Scenario: Sign is stable under small perturbation

- **WHEN** a bent trajectory's stages are perturbed by noise small relative to its extent, repeatedly
- **THEN** the returned orientation does not change sign across those perturbations

#### Scenario: Null configurations never approach antiparallel

- **WHEN** trajectories differing only by translation, uniform scale, or small noise are compared across repeated draws
- **THEN** no pairwise `angle` approaches 180 degrees

### Requirement: Genuine orientation differences are preserved

The estimator SHALL continue to report the true angle between trajectories that genuinely differ in direction, including differences exceeding 90 degrees, and SHALL reproduce the published reference outputs it was ported from.

#### Scenario: Known angle is recovered

- **WHEN** two trajectories are constructed with a known angle between their directions of progression
- **THEN** the reported `angle` equals that angle within numerical tolerance

#### Scenario: Reference outputs are reproduced

- **WHEN** the committed reference datasets are analyzed
- **THEN** the reported `angle` values match the committed reference results, including values greater than 90 degrees

### Requirement: The sign convention is documented as a deviation from the reference

The implementation SHALL record that the sign convention deviates deliberately from the reference supplement, which anchors on the raw first-stage row, and SHALL state why that anchor is unsuitable: it makes the orientation sign depend on the trajectory's position relative to the coordinate origin, so a pure translation can reverse it.

#### Scenario: Deviation is discoverable at the implementation

- **WHEN** a contributor reads the orientation estimator
- **THEN** it names the reference line it departs from, the reason, and the invariance the chosen anchor provides
