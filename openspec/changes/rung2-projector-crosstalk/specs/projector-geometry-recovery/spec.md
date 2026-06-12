## ADDED Requirements

### Requirement: Clean two-stage geometry injection with a selectable projector

The system SHALL provide a generator-free test bed that injects a *known* two-stage trajectory geometry directly in a clean linear (methylation M-value) feature space for two groups under multivariate-normal noise, and measures the trajectory `delta`/`angle` through a **selectable projector**. The test bed MUST reuse the Rung-0 step construction for the per-group steps (manipulations `none`, `magnitude`, `orientation`), MUST NOT apply the methylation `rev.logit` nonlinearity (the feature space is the M-value frame), and MUST hold the measurement path — `get_model_matrix`/`build_ls_means`, the two-group contrast, and `estimate_difference` — identical to Rung 0/1.

#### Scenario: Deterministic dataset from a seed

- **WHEN** the test bed is called with a fixed seed, feature dimension, sample sizes, noise/signal scales, a manipulation, and a projector selection
- **THEN** it returns measured `delta`/`angle` for the two-group contrast, and repeated calls with the same seed and projector return identical values

#### Scenario: Exactly two stages

- **WHEN** a dataset is generated
- **THEN** there are exactly two stages per group, so each group's trajectory is a single step vector and Procrustes shape is degenerate and excluded from all reported statistics

### Requirement: Four-projector comparison on identical geometry

The system SHALL expose the projector as an explicit parameter with at least four options: mean-centered **PCA** (the reference floor), per-feature **standardize** (matching the production `concat` integration transform), supervised **PLS-DA** (the production `stats/pls` projector, conditioned on the group label), and **SNF** spectral embedding (the production `stats/snf` graph-spectral projector). Each projector MUST map the same injected feature matrix to a latent matrix that is then measured by the identical estimator path, and the system MUST reuse the package's own `fit_plsda_transform`, `get_affinity_matrix`/`SNF`/`get_spectral` functions rather than reimplementing them.

#### Scenario: PCA arm reproduces the Rung-0 floor

- **WHEN** the projector is PCA on a small-effect `magnitude` and `orientation` manipulation
- **THEN** `magnitude` moves `delta` and `orientation` moves `angle`, with cross-talk near the `none` baseline — i.e. the Rung-0 clean-floor behavior is reproduced within tolerance

#### Scenario: Each projector measured on the same data

- **WHEN** the same seed and manipulation are run through PCA, standardize, PLS-DA, and SNF
- **THEN** the system reports each projector's measured `delta`/`angle` so they can be compared against the PCA floor on identical injected geometry

### Requirement: Per-projector distortion and cross-talk characterization

The system SHALL quantify, per projector and as a function of effect size and latent dimensionality, both the absolute distortion of the measured geometry against the intended geometry (magnitude target `signal_scale·(c−1)`, orientation target `θ`) and the cross-talk between facets (a `magnitude` manipulation's effect on `angle`, an `orientation` manipulation's effect on `delta`). Cross-talk MUST be read against the *same projector's* `none` null at the same settings, and the PCA floor MUST serve as the cross-projector reference.

#### Scenario: Cross-talk reported against the per-projector null and the PCA floor

- **WHEN** all manipulations have been run for a given projector over the seed set
- **THEN** the result reports, for that projector, the `magnitude`→`angle` and `orientation`→`delta` cross-talk relative to its own `none` floor, and the deviation of its `delta`/`angle` from the PCA floor, so projector-induced leakage is separated from finite-sample noise

#### Scenario: Dimensionality and effect-size sweep

- **WHEN** the test bed is run across a range of latent component counts and effect sizes
- **THEN** the result reports measured `delta`/`angle` and cross-talk as a function of both, so any onset of projector-induced distortion can be located

### Requirement: Supervised-leakage probe

The system SHALL provide a probe that detects whether the supervised PLS-DA projector, conditioned on the group label, manufactures spurious group-aligned geometry on a null trajectory. The probe MUST report the `none`-manipulation measured `delta`/`angle` under PLS-DA against the PCA floor, including a configuration that stresses leakage (e.g. larger component count relative to sample size).

#### Scenario: Null trajectory under a supervised projector

- **WHEN** the `none` manipulation (identical group trajectories) is measured under PLS-DA
- **THEN** the probe reports whether the measured `delta`/`angle` remains at the PCA `none` floor or is inflated by the group-conditioned projection, flagging label-induced spurious separation when present

### Requirement: SNF latent metric is reported as embedding-relative

The system SHALL report SNF spectral results primarily as recovery-versus-null (whether the injected geometry separates from the `none` manipulation), and MUST flag that absolute `delta`/`angle` magnitudes in the SNF embedding are embedding-relative — not unit-commensurable with the feature-space targets — rather than asserting a unit-matched distortion factor.

#### Scenario: SNF recovery without over-claimed magnitude

- **WHEN** a `magnitude` or `orientation` manipulation is measured under SNF
- **THEN** the result reports whether it separates from the `none` null and labels the absolute magnitude as embedding-relative, without claiming a numeric distortion ratio against the feature-space target

### Requirement: Findings writeup

The system SHALL produce a committed findings writeup that records, per projector, the distortion and cross-talk results (with the per-projector `none` floor and the PCA cross-projector reference), the effect-size and dimensionality sweeps, and the supervised-leakage probe, with enough parameter detail to reproduce the run, and SHALL state the gate decision for Rung 3 — which projector(s) are clean, whether the production `concat`/`snf` path leaks, and the next single factor (heterogeneous multi-omic concatenation or cross-omic coupling).

#### Scenario: Findings document exists and is reproducible

- **WHEN** the test bed and sweeps have been run
- **THEN** a findings document records the per-projector distortion/cross-talk table, the sweeps, the leakage probe, the Rung-3 gate decision, and the exact parameters (seed set, dimensions, noise/signal scale, projector settings, component grid) needed to reproduce them
