## Rung Ladder: Conclusion

This document closes the rung ladder investigation into the magnitude→orientation cross-talk observed in the specificity study.

---

## What the ladder tested

The ladder added one structural factor per rung to a clean linear baseline, measuring whether each factor could produce the observed cross-talk (magnitude manipulation leaking into the angle statistic).

| Rung | Module | Factor tested | Finding |
|-----:|--------|---------------|---------|
| 0 | `linear_recovery` | Linear geometry recovery via PCA | Clean. Background angular scatter (k-noise floor) documented. |
| 1 | `methylation_recovery` | B-value vs M-value methylation (sigmoid nonlinearity) | **B values distort step geometry.** M-value integration correctly inverts the transform. |
| 2 | `projector_recovery` | Per-feature standardisation vs integration projector | Standardisation clean. **SNF leaks** independently. |
| 3 | `multiblock_recovery` | Multi-block concatenation with independent nuisance blocks | Clean at all dimensionality ratios and intra-block correlations. |
| 4 | `coupling_recovery` | Cross-omic coupling (linear mean shift via incidence matrix M) | Clean. MᵀM distortion negligible at production sparsity. |

---

## Conclusion

**The cross-talk is an input-space artifact, not an estimator artifact.**

The primary mechanism is the sigmoid nonlinearity introduced when methylation data is left in B-value space instead of M-value space:

1. The generator constructs group trajectories in M-value space, where the magnitude manipulation is `b = c·a` — a pure scaling that is direction-preserving by construction.
2. After the `inv_logit` transform to B values, `sigmoid(c·a) ≠ c·sigmoid(a)`. The unequal squeeze near 0 and 1 — which depends on each group's baseline — converts a parallel trajectory in M-value space into a diverging trajectory in B-value space. Magnitude and orientation are no longer separable in the input.
3. Any downstream estimator (concat, SNF, PLS) running on B-value methylation will inherit this manufactured cross-talk regardless of how clean its linear algebra is.

The secondary factor is a **misattribution of source**: running the specificity study through `concat` and observing cross-talk leads to the easy but incorrect conclusion that the estimator is broken. Rungs 0, 3, and 4 collectively show concat is clean under ideal (linear, properly-scaled) inputs — the contamination is in the input space, not the measurement.

---

## What remains open

**SNF leaks independently (Rung 2).** The SNF cross-talk is real but is a separate story from the B-value artifact — its source is the nonlinear affinity kernel and graph diffusion, not input scaling. This is not a blocker for the current study (PLS is the production latent-space method) but warrants its own investigation if SNF results are interpreted.

**Production-scale coupling was not tested.** Rung 4 used synthetic M at p_anchor=50. Real InterSIM dimensionalities (p_cpg~27,000, p_gene~600) could in principle alter the MᵀM spectrum, though the analytic formula suggests the effect would remain small at realistic sparsity. Not a priority given the B-value finding.

---

## Immediate implication

Ensure the production pipeline feeds **M-value methylation** (not B values) into the integration step. With M values, the concat path is clean (Rungs 0, 3, 4), the nonlinearity is removed (Rung 1), and the specificity-study cross-talk should not be present. The PLS integration path is additionally robust because the double cross-validation stage constructs the latent space from the stage label directly, further suppressing any residual scale effects.