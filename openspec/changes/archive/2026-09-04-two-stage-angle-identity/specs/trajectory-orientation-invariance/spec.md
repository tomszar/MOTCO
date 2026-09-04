## ADDED Requirements

### Requirement: Two-stage pairwise angle equals the transition-vector angle

For trajectories with exactly two stages, the pairwise `angle` statistic SHALL equal the angle between the two groups' unit transition vectors (first stage toward second stage). This is an identity, not an approximation: with two stages the centered stage configuration is rank one, its leading principal axis is the transition direction, and the net-displacement sign anchor orients it along the progression. Agreement MUST hold at machine precision, assessed on the cosine of the angle (the arccos map is ill-conditioned near 0 and 180 degrees, so the tolerance contract is stated on the cosine). Trajectories whose two stages coincide are the documented zero-displacement degeneracy and are outside this requirement.

#### Scenario: Synthetic two-stage angle is the transition-vector angle

- **WHEN** two two-stage trajectories with distinct stages are compared via the pairwise `angle` statistic
- **THEN** the cosine of the reported angle equals the inner product of the two unit transition vectors within machine precision

#### Scenario: Example1 fixture angles equal the direct-vector angles

- **WHEN** the committed two-stage reference dataset (`evo_649_sm_example1.csv`) is analyzed
- **THEN** every pairwise `angle` equals the angle computed directly from that pair's unit transition vectors, derived from the same fitted stage means, within machine precision on the cosine

#### Scenario: Committed R angles are the sign-anchor artifact of the direct-vector angles

- **WHEN** the reported two-stage angles are compared against the committed R reference outputs for the two-stage dataset
- **THEN** each committed angle equals either the direct-vector angle or its supplement (180 degrees minus it), and the supplement cases are attributable to the reference's raw first-stage sign anchor

### Requirement: Two-stage regression expectations are pinned to the progression convention

Regression tests that compare two-stage `angle` outputs against committed reference values MUST NOT accept both an angle and its supplement. The expected value SHALL be the direct transition-vector angle under the progression sign convention, so that a sign-flip regression in the orientation estimator fails the comparison rather than passing through a supplementary-angle acceptance.

#### Scenario: A sign flip is not absorbed by supplementary acceptance

- **WHEN** a two-stage regression comparison is evaluated against its expected angle
- **THEN** an output equal to the supplement of the expected angle fails the comparison
