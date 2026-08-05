## Context

MOTCO currently studies the forward relationship from feature-space interventions to trajectory statistics measured after integration. This change adds the complementary diagnostic: fit a simple two-component PLS-DA representation, impose a known trajectory change in its score space, and reconstruct the feature change implied by that fixed linear model.

The study is intentionally generator-independent. It is meant to confirm linear-geometric expectations, not reproduce the production semi-synthetic pipeline. The existing `fit_plsda_model` helper returns a fitted scikit-learn `PLSRegression`, but the study needs explicit access to scores, X loadings, internal centering/scaling, and inverse transformation. It can use the fitted estimator directly without changing the public PLS API.

## Goals / Non-Goals

**Goals:**

- Produce deterministic two-stage and three-stage Gaussian feature datasets with two exactly identical groups before intervention.
- Fit one pooled, frozen, two-component PLS-DA model conditioned on stage.
- Impose controlled magnitude, orientation, and three-stage bend interventions on only Group B's latent trajectory.
- Preserve within-stage latent residuals and feature variation outside the retained PLS representation.
- Quantify the intended latent geometry, reconstructed feature geometry, per-feature changes, loading alignment, round-trip accuracy, and the metric induced by the PLS inverse map.
- Keep the study small, reproducible, and independently testable.

**Non-Goals:**

- No InterSIM or semi-synthetic multi-omics generator, omics blocks, biological cascade, or methylation transform.
- No double cross-validation, component selection, RRPP, p-values, power curves, or production tuning.
- No SNF/PCA comparison and no change to MOTCO's production PLS integration.
- No claim that the reconstructed feature perturbation is a unique or biologically causal inverse.
- No requirement that the free three-stage bend preserve total path length.

## Decisions

### 1. Paired duplication gives an exact null baseline

Generate stage-conditioned Gaussian samples once and duplicate every sample into Groups A and B. Before intervention, their feature matrices, PLS scores, and stage centroids are exactly equal. The study compares Group B before and after intervention, so all detected differences are attributable to the controlled score displacement.

Alternative considered: independently sample equivalent groups. Rejected for the initial study because finite-sample group differences obscure the inverse mapping without adding information about its linear geometry.

### 2. Use a single generic feature matrix and exactly two PLS components

Use a small matrix such as 50 features with equal samples per stage and isotropic Gaussian noise around deterministic stage means. Fit one `PLSRegression(n_components=2, scale=True)` through the existing PLS-DA encoding path, with stage as the response. Two fixed components provide one common intervention plane for both stage counts and remove CV-selected dimensionality as a confound.

Alternative considered: production multi-omics preprocessing and double-CV sizing. Rejected because this study isolates the inverse geometry rather than production performance.

### 3. Freeze one pooled PLS model

Fit PLS once on the unmodified pooled A+B dataset. All original scores, modified scores, inverse reconstructions, and round-trip transformations use that same estimator. The model is never refit after an intervention, since refitting would change the coordinate system and confound the imposed geometry with projector movement.

### 4. Move stage centroids while preserving within-cell residuals

For each Group B sample `i` in stage `s`, construct

```
T_star[i] = T[i] - centroid_B[s] + centroid_B_star[s]
```

so within-stage score residuals and covariance are unchanged.

- **Magnitude:** scale all stage centroids around their trajectory centroid by a fixed factor `c` (headline `c=2`).
- **Orientation:** rigidly rotate all centered centroids in the LV1-LV2 plane by a fixed angle `theta` (headline `theta=45°`).
- **Shape:** for three stages only, hold the endpoints fixed and move the middle centroid by `d*q`, where `q` is perpendicular to the endpoint axis (headline `d` expressed as a fixed fraction of endpoint distance). This is a controlled bend, not a promise of zero magnitude/orientation co-movement.

The midpoint/trajectory centroid remains fixed for magnitude and orientation, preventing an unintended translation effect.

### 5. Reconstruct changes additively

PLS inverse transformation reconstructs only the part of X represented by retained components. Replacing X with `inverse_transform(T_star)` would discard the original residual. Instead compute

```
delta_X = inverse_transform(T_star) - inverse_transform(T)
X_star = X + delta_X
```

for Group B. Group A remains unchanged. This preserves each original feature residual while adding exactly the change implied by the fixed PLS model and its fitted scaling.

The study records the round-trip error between the requested score displacement and the score displacement recovered by transforming `X_star`.

### 6. Report geometry in both spaces without inference

For each stage count and intervention, record the group-pair `delta`, `angle`, and (when defined) Procrustes `shape` statistics before and after reconstruction in:

- the two-dimensional PLS score space; and
- the original feature space.

The latent results verify the intervention, while the feature results reveal which geometric properties survive the inverse map. No permutations or rejection rates are computed.

### 7. Characterize the feature perturbation and induced metric

Return a tidy per-feature table containing signed/absolute changes, PLS loadings, and ranks, plus aggregate summaries including non-negligible feature count and correlation between absolute change and loading magnitude.

For the score-to-feature map `x = t P^T`, report the two-dimensional induced metric

```
G = P^T P
```

and its eigenvalues/condition number. `G` proportional to the identity means score-space rotations are feature-space isometries up to a common scale; anisotropy predicts direction-dependent length and angle distortion.

### 8. Outputs are reproducible and findings-ready

Provide a driver with fixed defaults and explicit seed/configuration options. It writes a machine-readable cell-level summary and a compact Markdown report containing parameters, geometry comparisons, reconstruction diagnostics, and leading changed features. The module returns structured dataclasses or equivalent typed records so tests do not parse presentation text.

## Risks / Trade-offs

- [Exact duplicate samples understate real-data uncertainty] → Label the study as a deterministic inverse-geometry diagnostic; sampling robustness is a separate follow-up.
- [PLS inverse is not a unique inverse of the original features] → Use additive reconstruction, preserve original residuals, and describe results as the feature perturbation implied by the fitted PLS representation.
- [PLS loadings are generally dense] → Quantify concentration rather than expecting sparse feature changes.
- [A three-stage bend can change path length and principal orientation] → Report all three statistics and call it a controlled bend rather than a strictly isolated shape effect.
- [Internal PLS scaling can make manual loading formulas easy to misuse] → Use the estimator's transform/inverse-transform methods for reconstruction and validate with a round-trip diagnostic; use loadings only for analytic interpretation.
- [Degenerate simulated stage geometry could make rotation or the perpendicular direction unstable] → Construct separated deterministic stage means and validate nonzero endpoint distance before intervention.

## Migration Plan

This change is additive and requires no migration or rollback. Removing the new simulation module, driver, and tests restores the previous repository state without affecting public behavior.

## Open Questions

- Choose the final default sample count, Gaussian noise scale, and shape displacement fraction during implementation while keeping them exposed in the study parameter object.
- Decide whether the Markdown report should include a small plot; tabular output is sufficient for the required capability and visualization is optional.
