## 1. Study data and model

- [x] 1.1 Add a typed parameter/result model for the inverse-PLS study, including seed, stage count, feature count, samples per stage, noise, and intervention strengths.
- [x] 1.2 Implement deterministic Gaussian stage data and exact paired duplication into Groups A and B.
- [x] 1.3 Fit one frozen two-component PLS-DA model to the pooled stage labels and expose its scores, loadings, and fitted preprocessing state to the study.
- [x] 1.4 Add tests proving the baseline groups have identical features, scores, centroids, and trajectory geometry.

## 2. Latent interventions

- [x] 2.1 Implement centroid magnitude scaling about the trajectory centroid while preserving within-stage score residuals.
- [x] 2.2 Implement rigid LV1-LV2 centroid rotation about the trajectory centroid while preserving pairwise latent distances and within-stage residuals.
- [x] 2.3 Implement the three-stage shape bend with fixed endpoints and a perpendicular middle-centroid displacement.
- [x] 2.4 Add numerical tests for each intervention's defining invariants and reject shape for fewer than three stages.

## 3. Reconstruction and diagnostics

- [x] 3.1 Implement additive PLS reconstruction as the inverse-transform difference added to original Group B features, leaving Group A unchanged.
- [x] 3.2 Implement and test the transform-after-reconstruction round-trip diagnostic and preservation of the original feature residual.
- [x] 3.3 Compute latent-space and feature-space `delta`, `angle`, and available `shape` summaries for every study cell.
- [x] 3.4 Produce tidy per-feature signed/absolute changes with component loadings, ranks, concentration summaries, and loading-change association.
- [x] 3.5 Compute `G = P^T P`, its eigenvalues and condition number, and test the isotropic versus anisotropic interpretation on controlled loading matrices.

## 4. Driver and reporting

- [x] 4.1 Add a reproducible driver covering two-stage magnitude/orientation and three-stage magnitude/orientation/shape with explicit configuration arguments.
- [x] 4.2 Write machine-readable cell and feature results plus a compact Markdown report containing parameters, geometry comparisons, reconstruction errors, induced-metric diagnostics, and leading changed features.
- [x] 4.3 Add driver/output-schema tests and a small smoke run that completes with deterministic results.

## 5. Verification and findings

- [x] 5.1 Run focused inverse-study tests and the fast repository test suite.
- [x] 5.2 Run Ruff and mypy on the added module, tests, and driver and address all introduced findings.
- [x] 5.3 Run the default study and add a findings document that distinguishes mathematical expectations from observed numerical results and states the non-unique, non-causal interpretation of PLS inversion.
