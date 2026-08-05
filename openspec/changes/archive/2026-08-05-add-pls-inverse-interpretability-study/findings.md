# Inverse PLS trajectory interpretability: findings

## Scope

This is a deterministic linear diagnostic, not a power study. Gaussian stage samples were duplicated exactly into Groups A and B, one frozen two-component PLS-DA model was fitted to stage, and only Group B's stage centroids were modified in score space. The feature counterfactual was reconstructed additively so variation outside the two retained components remained unchanged.

Default run: seed 0, 50 features, 30 samples per stage per group, noise scale 0.5, signal scale 4.0, magnitude factor 2, orientation rotation 45 degrees, and shape displacement equal to 0.5 times the endpoint distance.

## Results

| stages | intervention | latent delta | latent angle | latent shape | feature delta | feature angle | feature shape |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2 | magnitude | 6.4250 | 0.0000 | — | 4.1810 | 0.7874 | — |
| 2 | orientation | 0.0000 | 45.0000 | — | 0.0665 | 42.0754 | — |
| 3 | magnitude | 8.8015 | 0.0000 | 0.0000 | 5.1989 | 1.6769 | 0.0484 |
| 3 | orientation | 0.0000 | 45.0000 | 0.7654 | 0.0112 | 42.0745 | 0.7506 |
| 3 | shape bend | 2.2721 | 2.2085 | 0.4550 | 1.2032 | 1.9986 | 0.4739 |

The requested score displacements survived the inverse/forward round trip to machine precision: maximum errors were `1.3e-15` to `2.7e-15`. Feature residual preservation errors were likewise at numerical zero. The implementation therefore performs the intended additive counterfactual rather than replacing samples with their low-rank PLS reconstructions.

## Interpretation

### 1. Magnitude is exact in the represented PLS component, but only approximate in the complete feature matrix

Scaling the centered score trajectory is linear, so its PLS-represented feature component scales exactly. The final counterfactual also retains the original feature residual, including its stage-dependent mean contribution. That residual is not scaled. Consequently the complete feature trajectory shows small orientation/shape co-movement (0.79–1.68 degrees; three-stage shape 0.048) rather than an exact global scaling.

This is the intended consequence of additive reconstruction: it isolates the perturbation implied by PLS while preserving information that PLS omitted.

### 2. A 45-degree latent rotation becomes an approximately 42-degree feature rotation

Orientation preserved latent pairwise distances exactly and introduced effectively zero latent `delta`. After reconstruction, the feature angle was about 42.1 degrees with very small `delta` (0.066 for two stages and 0.011 for three). PLS inversion therefore retained the dominant interpretation while introducing modest distortion from the loading metric, fitted scaling, and preserved residual.

The standardized-loading metric was mildly anisotropic for two stages (condition number 1.32) and almost isotropic for three stages (1.03). This is consistent with greater two-stage distortion, while also showing that `P^T P` alone does not describe the complete original-unit geometry when fitted scaling and unrepresented residuals are retained.

### 3. The shape statistic responds to a known rigid rotation before reconstruction

The orientation intervention is a rigid rotation in the two-dimensional score plane: the implementation test confirms that all latent pairwise stage distances are unchanged. Nevertheless, MOTCO reports latent `shape=0.765` for the three-stage rotated trajectory and feature `shape=0.751` after reconstruction.

The response is therefore not created by the inverse PLS map—the nonzero value is already present in the intentionally rigid latent intervention. This reinforces the earlier finding that the current Procrustes `shape` statistic acts as a broad geometry detector and can co-move with orientation. It warrants separate estimator-specific investigation if strict rotation invariance is required.

### 4. The free middle-stage bend is intentionally not shape-only

Moving the middle point perpendicular to the fixed endpoint axis produced the intended latent shape difference (0.455), but also changed path length (`delta=2.272`) and principal orientation (2.21 degrees). Feature reconstruction retained a comparable shape response (0.474) with smaller but nonzero magnitude and orientation changes. These are properties of the simple free-bend construction, not reconstruction failures.

### 5. Latent interventions imply dense, loading-aligned feature changes

All 50 features received non-negligible changes. The effective participation ratio was about 31–37 features, and the five largest-change features (top 10%) carried about 22–27% of total absolute change. Loading/change correlations ranged from 0.26 for two-stage magnitude to 0.92 for three-stage orientation.

Thus, a simple movement in PLS space generally represents a dense low-rank molecular pattern, not a sparse feature edit. Orientation and three-stage magnitude were especially well aligned with overall loading magnitude; the weaker two-stage magnitude correlation reflects the particular direction-specific combination of the two loading columns.

## Conclusion

The study confirms the main expectation: controlled PLS-score interventions map reproducibly and almost geometrically faithfully into feature changes through a fixed linear inverse. The remaining discrepancies are interpretable consequences of a non-isometric loading/scaling map plus preservation of feature variation outside the retained PLS components.

The reconstructed perturbation is **not a unique inverse and not a biologically causal counterfactual**. It is the dense additive feature pattern implied by this fitted two-component PLS model under the chosen residual-preservation convention.

## Forward interpretation of an observed orientation difference

The reverse study supports, but refines, the original feature-surgery intuition. In a fixed linear PLS model, a latent orientation difference means that the groups use different relative combinations of the stage-associated feature patterns retained by PLS. This can arise from different active feature sets, but also from overlapping features with different relative magnitudes or signs.

For a two-stage production comparison, define standardized, preferably covariate-adjusted, feature transitions `d_A` and `d_B`. The feature-level directional contrast is

```text
q = d_B / ||d_B|| - d_A / ||d_A||.
```

Ranking `|q_j|` surfaces features whose relative contribution to progression differs most between groups while removing overall trajectory magnitude. For multi-stage data, the same comparison can be made per adjacent transition and between the groups' signed principal feature-space orientations.

PLS-based attribution should reconstruct each group's stage-to-stage score displacement through the same frozen pooled loadings and compare the resulting normalized feature patterns. Comparing this reconstruction with the directly observed feature contrast separates the group difference captured by retained PLS components from the feature-space residual discarded by the projection. Bootstrap rank/sign stability and module-level summaries are necessary because correlated features can exchange individual loading weight without changing the underlying multivariate pattern.

VIP scores are not sufficient attribution by themselves: pooled VIP identifies importance for stage prediction, whereas orientation drivers are features whose **normalized stage effects differ between groups**.
