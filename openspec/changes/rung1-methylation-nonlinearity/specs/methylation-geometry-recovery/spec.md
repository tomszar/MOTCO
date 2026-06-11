## ADDED Requirements

### Requirement: Methylation-only two-stage test bed with rev.logit

The system SHALL provide a generator-free test bed that injects a *known* two-stage trajectory geometry directly in methylation **M-value (logit) space** for two groups under isotropic multivariate-normal noise, and then applies InterSIM's inverse-logit (`rev_logit`, reused from the generator) to obtain methylation β values that are measured. The test bed MUST reuse the Rung-0 step construction for the per-group steps and MUST NOT generate or measure an expression or protein layer, MUST NOT invoke the cross-omic cascade, and MUST NOT reimplement the inverse-logit.

#### Scenario: Deterministic β dataset from a seed

- **WHEN** the test bed is called with a fixed seed, feature dimension, sample sizes, noise/signal scales, a manipulation, and an operating-point baseline
- **THEN** it returns a β matrix in the open interval (0, 1) and group/stage labels whose per-(group, stage) β-sample means approximate `rev_logit` of the specified M-space configuration, and repeated calls with the same seed return identical matrices

#### Scenario: Exactly two stages

- **WHEN** a dataset is generated
- **THEN** there are exactly two stages per group, so each group's trajectory is a single step vector and Procrustes shape is degenerate and excluded from all reported statistics

### Requirement: Operating point as the swept independent variable

The system SHALL expose the methylation baseline operating point (an M-value offset placing the trajectory on the `rev_logit` sigmoid) as an explicit parameter, and SHALL provide a sweep over it from the sigmoid center (approximately linear) to the saturating tails. For each swept operating point the system MUST report the measured `delta` and `angle` for the `none`, `magnitude`, and `orientation` manipulations, with the `none` manipulation calibrating the null floor at that operating point.

#### Scenario: Center operating point reduces to the linear floor

- **WHEN** the baseline operating point is at the sigmoid center and the step is small
- **THEN** the measured `delta`/`angle` for `magnitude`/`orientation` match the Rung-0 clean-floor behavior (magnitude moves `delta`, orientation moves `angle`, cross-talk near the `none` baseline) within tolerance

#### Scenario: Saturating operating point distorts recovery

- **WHEN** the baseline operating point is moved into the saturating tail of the sigmoid
- **THEN** the measured β-space `delta` for the `magnitude` manipulation is compressed relative to the intended M-space magnitude, demonstrating that the `rev_logit` nonlinearity is exercised

### Requirement: Distortion and cross-talk characterization

The system SHALL quantify, as a function of operating point, both the absolute distortion of the measured β-space geometry against the intended M-space geometry (the magnitude target `signal_scale·(c−1)` and the orientation target `θ`) and the cross-talk between facets (a `magnitude` manipulation's effect on `angle`, and an `orientation` manipulation's effect on `delta`), read against the `none` null floor at the same operating point.

#### Scenario: Cross-talk is reported against the null floor

- **WHEN** the operating-point sweep has been run for all three manipulations
- **THEN** the result reports, per operating point, the `magnitude`→`angle` and `orientation`→`delta` cross-talk relative to the `none` baseline at that operating point, so any leakage attributable to `rev_logit` is separated from finite-sample noise

### Requirement: First-order inverse design at the operating point

The system SHALL provide a first-order (Jacobian-linearized) inverse design that maps a target β-space latent step back to an M-value-space feature change, using the local map `J = diag(β(1−β))` at the operating point composed with the Rung-0 linear inverse-design formula. The result MUST be labeled first-order and MUST expose the M-space change's support and sparsity summary for inspection.

#### Scenario: Linearized round-trip near the center

- **WHEN** a target latent step is requested at an operating point near the sigmoid center
- **THEN** the first-order inverse design returns an M-space change whose linearized projection recovers the requested latent target within tolerance, and its support/sparsity summary is reported with the first-order caveat noted

### Requirement: Findings writeup

The system SHALL produce a committed findings writeup that records the operating-point sweep (measured `delta`/`angle` and cross-talk versus baseline, with the `none` floor), the secondary step-scale sweep, and the absolute-distortion summary, with enough parameter detail to reproduce the run, and SHALL state the gate decision for Rung 2 — whether the `rev_logit` nonlinearity alone reproduces the specificity-study cross-talk and at what operating point it onsets.

#### Scenario: Findings document exists and is reproducible

- **WHEN** the test bed and sweeps have been run
- **THEN** a findings document records the operating-point and step-scale results, the distortion/cross-talk summary, the Rung-2 gate decision, and the exact parameters (seed set, dimensions, noise/signal scale, operating-point grid, number of PCA components) needed to reproduce them
