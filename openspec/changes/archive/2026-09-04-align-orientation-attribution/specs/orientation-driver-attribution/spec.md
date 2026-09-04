## ADDED Requirements

### Requirement: Attribution decomposes the tested principal-orientation contrast

The system SHALL compute, for each group, the principal orientation of its centered stage-mean configuration in feature space — the leading principal axis, signed so it points along the trajectory's net displacement from first stage to last, using the same convention as the tested `angle` estimator — for both the observed group-stage means and their PLS-reconstructed counterparts. The system SHALL report the signed per-feature principal-orientation contrast between the two ordered groups as observed, PLS-captured, and residual (observed minus captured) components, beside the existing per-transition contrasts. The group ordering and sign conventions SHALL be identical to those used for transition contrasts.

#### Scenario: Principal contrast is estimator-consistent

- **WHEN** the principal orientation is computed from a group's observed stage means
- **THEN** it equals, within numerical tolerance, the orientation the trajectory `angle` estimator returns for the same stage-mean configuration

#### Scenario: Two-stage principal contrast equals the transition contrast

- **WHEN** the design contains exactly two stages and both groups have nonzero transitions
- **THEN** the observed, PLS-captured, and residual principal-orientation contrasts equal the corresponding single-transition contrasts exactly

#### Scenario: Multi-stage principal contrast is reported beside transitions

- **WHEN** the design contains three or more ordered stages
- **THEN** the result contains one principal-orientation contrast per attribution component in addition to the per-adjacent-transition contrasts, in the same structured and tabular forms

### Requirement: Principal-orientation degeneracy is qualified, not silently reported

The system SHALL report, for each group and for both the observed and PLS-reconstructed configurations, the relative eigengap of the centered stage-mean configuration, and SHALL mark the principal-orientation contrast as degenerate when any contributing configuration's relative eigengap falls below a documented threshold. A degenerate principal contrast SHALL remain available in the output but MUST carry an explicit degeneracy flag; the system MUST NOT present a near-isotropic configuration's principal axis as a well-defined orientation.

#### Scenario: Well-separated configuration passes the gate

- **WHEN** every contributing stage-mean configuration has a relative eigengap at or above the documented threshold
- **THEN** the principal-orientation contrast is reported without a degeneracy flag and the per-group eigengaps are included in the result

#### Scenario: Near-isotropic configuration is flagged

- **WHEN** a contributing stage-mean configuration has a relative eigengap below the documented threshold
- **THEN** the principal-orientation contrast carries a degeneracy flag identifying which group and configuration triggered it

#### Scenario: Closed trajectory is unavailable

- **WHEN** a group's net displacement from first to last stage is zero or below the configured numerical tolerance
- **THEN** the principal orientation for that group is marked unavailable, mirroring the zero-transition handling

### Requirement: Bootstrap stability covers the principal-orientation contrast

When bootstrap resampling is enabled, the system SHALL summarize feature sign stability, rank stability, and top-k selection frequency for the principal-orientation contrast with the same stratified resampling contract as transition contrasts, and SHALL anchor each replicate's principal-axis sign by that replicate's net displacement so resampling noise cannot flip the axis arbitrarily.

#### Scenario: Principal-orientation bootstrap is reproducible

- **WHEN** attribution is run twice with the same data, bootstrap count, and seed
- **THEN** the principal-orientation stability summaries are identical

#### Scenario: Replicate sign is anchored

- **WHEN** a bootstrap replicate's stage-mean configuration is resampled
- **THEN** its principal axis is signed by that replicate's own net displacement before contrasting, using the same convention as the point estimate

## MODIFIED Requirements

### Requirement: Attribution output is machine-readable and interpretation-bounded

The system SHALL return structured results that retain group and stage ordering, transition identifiers, feature names, model/component metadata, effective bootstrap parameters, and unit metadata. The output SHALL state that attribution describes associations implied by the fitted shared PLS representation and SHALL not claim a unique inverse, biological causality, or inferential significance. The interpretation metadata SHALL state which quantity each decomposition explains: the principal-orientation contrast decomposes the tested `angle` estimand (principal-axis divergence), while per-adjacent-transition contrasts describe where along the trajectory the group directions diverge and coincide with the tested estimand only for two-stage designs.

#### Scenario: Result can be consumed without parsing prose

- **WHEN** a valid attribution analysis completes
- **THEN** feature-level, transition-level, principal-orientation, aggregate, and configuration data are available as structured records or tables

#### Scenario: Significance is not fabricated

- **WHEN** attribution is run without an external hypothesis-test result
- **THEN** the result contains no synthetic p-value or significance decision and leaves selection of significant global results to the calling workflow

#### Scenario: Decomposition targets are stated

- **WHEN** a valid attribution analysis completes
- **THEN** the interpretation metadata distinguishes the principal-orientation contrast as the decomposition of the tested `angle` estimand from the per-transition contrasts as a per-step description
