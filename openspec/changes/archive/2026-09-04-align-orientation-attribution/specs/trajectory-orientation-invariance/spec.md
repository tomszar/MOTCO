## MODIFIED Requirements

### Requirement: Genuine orientation differences are preserved

The estimator SHALL continue to report the true angle between trajectories that genuinely differ in principal-axis orientation, including differences exceeding 90 degrees, and SHALL reproduce the published reference outputs it was ported from. Documentation of the `angle` statistic SHALL state its estimand as principal-axis divergence: the angle between the groups' leading principal axes, each signed by net displacement — which equals the angle between directions of progression only for straight or two-stage trajectories.

#### Scenario: Known angle is recovered

- **WHEN** two trajectories are constructed with a known angle between their signed principal axes
- **THEN** the reported `angle` equals that angle within numerical tolerance

#### Scenario: Reference outputs are reproduced

- **WHEN** the committed reference datasets are analyzed
- **THEN** the reported `angle` values match the committed reference results, including values greater than 90 degrees

#### Scenario: Estimand is documented as principal-axis divergence

- **WHEN** a reader consults the orientation documentation for a multi-stage trajectory
- **THEN** it states that `angle` compares signed principal axes, not per-step transition directions, and that the two coincide only for straight or two-stage trajectories
