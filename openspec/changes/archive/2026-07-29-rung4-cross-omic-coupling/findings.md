## Summary

Cross-omic coupling as modeled by a linear mean shift (`coupling_scale × M_norm @ anchor_step`) produces **no magnitude→orientation cross-talk** and **no orientation distortion** at any coupling strength or M structure. The analytic formula confirms that the MᵀM inner-product perturbation is negligible at production-realistic sparsity (nnz=3/50), and the magnitude arm is analytically and empirically direction-preserving. **Cross-omic coupling at the feature-mean level is not the source of the specificity-study cross-talk.**

Gate: **Rung 5 = SNF mechanism isolation.** All single factors on the `concat` path are now cleared (Rungs 0–4). Rung 2 established SNF leaks; the rung ladder's remaining question is which mechanism within SNF produces the cross-talk.

---

## Axis 1 — Coupling sweep

**random_sparse M (production-realistic headline)**

| coupling_scale | none θ | magnitude θ | orientation θ |
|---------------:|-------:|------------:|--------------:|
| 0.00 | 17.4° | 16.4° | 51.6° |
| 0.25 | 17.2° | 16.2° | 51.5° |
| 0.50 | 16.6° | 15.5° | 51.3° |
| 0.75 | 15.8° | 14.7° | 51.0° |
| 1.00 | 15.0° | 13.8° | 50.8° |

`magnitude` angle tracks the `none` null floor exactly across every coupling_scale — no cross-talk.  
`orientation` angle is stable at ~51° (the k-noise PCA floor from Rung 0) and does not increase with coupling — the coupling does not distort the measured orientation.

**dense M (maximum-density upper bound)**

| coupling_scale | none θ | magnitude θ | orientation θ |
|---------------:|-------:|------------:|--------------:|
| 0.00 | 17.4° | 16.4° | 51.6° |
| 0.25 | 17.4° | 16.4° | 51.5° |
| 0.50 | 17.3° | 16.3° | 51.4° |
| 0.75 | 17.1° | 16.2° | 51.2° |
| 1.00 | 17.0° | 16.0° | 51.0° |

Even with a fully dense M (maximum possible `MᵀM` distortion), the effects are sub-degree and track the null floor.

**rank1 M**

| coupling_scale | none θ | magnitude θ | orientation θ |
|---------------:|-------:|------------:|--------------:|
| 0.00 | 17.4° | 16.4° | 51.6° |
| 0.50 | 17.3° | 16.4° | 51.6° |
| 1.00 | 17.2° | 16.3° | 51.5° |

Smallest effects of all three structures. Rank-1 M projects only a single anchor feature into nuisance space; the coupling contribution is near zero for a random step vector.

**Null floor decreases with coupling_scale (random_sparse).** Adding coupled signal in the nuisance block provides additional stage information in the joint PCA, reducing the background angular scatter. This is the opposite of cross-talk — coupling slightly improves null-level specificity.

---

## Axis 2 — Analytic vs PCA-measured angle (orientation arm)

**random_sparse M**

| coupling_scale | θ_pred (analytic) | θ_meas (PCA) | θ_anchor |
|---------------:|------------------:|-------------:|---------:|
| 0.00 | 45.00° | 51.6° | 51.6° |
| 0.25 | 45.00° | 51.5° | 51.6° |
| 0.50 | 45.00° | 51.3° | 51.6° |
| 0.75 | 45.01° | 51.0° | 51.6° |
| 1.00 | 45.02° | 50.8° | 51.6° |

The analytic formula predicts **essentially zero distortion** (45.00°–45.02° across coupling_scale 0–1) for random_sparse M with nnz=3/50. The feature-space inner product is barely perturbed by the `MᵀM` term at this sparsity. The gap between `θ_pred ≈ 45°` and `θ_meas ≈ 51°` is the PCA k-noise floor documented in Rung 0, not coupling-induced.

**dense M**

| coupling_scale | θ_pred (analytic) | θ_meas (PCA) |
|---------------:|------------------:|-------------:|
| 0.00 | 45.00° | 51.6° |
| 0.50 | 44.93° | 51.4° |
| 1.00 | 44.72° | 51.0° |

Dense M produces slightly larger analytic distortion (−0.28° at coupling_scale=1), but in the direction of **decreasing** the measured angle, not increasing it. The `MᵀM` for the all-ones matrix aligns all coupling energy onto the mean direction of the anchor step, which slightly collapses the angular separation — the opposite of the specificity-study cross-talk (which manifests as angle *increasing* for `magnitude`).

**rank1 M**

| coupling_scale | θ_pred (analytic) | θ_meas (PCA) |
|---------------:|------------------:|-------------:|
| 0.00 | 45.00° | 51.6° |
| 0.50 | 44.96° | 51.6° |
| 1.00 | 44.86° | 51.5° |

Negligible in both directions.

**Mechanism confirmed, mechanism is negligible.** The analytic formula correctly predicts the feature-space coupling effect: for production-realistic M sparsity (3 non-zeros per 50 anchor features), coupling is negligible. The cross-talk source is not the MᵀM inner-product distortion.

---

## Axis 3 — dim_ratio sweep at coupling_scale=0.75 (random_sparse)

| dim_ratio | none θ | magnitude θ | orientation θ |
|----------:|-------:|------------:|--------------:|
| 0.5 | 16.1° | 13.8° | 50.6° |
| 1.0 | 15.8° | 14.7° | 51.0° |
| 5.0 | 21.5° | 21.0° | 50.0° |

At dim_ratio=5 (nuisance block 5× larger), the null floor rises to 21.5° (k-noise dilution, same as Rung 3). The `magnitude` arm tracks the null floor exactly (21.0° ≈ 21.5°). `orientation` stays near 50°. No coupling-induced cross-talk at any dimensionality ratio.

---

## Axis 4 — Matrix-seed stability (coupling_scale=0.75, random_sparse)

| matrix_seed | none θ | magnitude θ | orientation θ |
|------------:|-------:|------------:|--------------:|
| 0 | 15.8° | 14.7° | 51.0° |
| 1 | 15.7° | 14.7° | 50.6° |
| 2 | 16.2° | 15.2° | 50.8° |
| 3 | 15.9° | 14.9° | 51.2° |
| 4 | 16.2° | 15.2° | 50.6° |

`magnitude` tracks the null floor consistently across all matrix seeds. `orientation` is stable at ~51°. Results are not matrix-seed-dependent.

---

## Interpretation

**Why coupling does not produce cross-talk:**

1. **Magnitude arm is direction-preserving analytically.** For `b = c·a`, the joint steps `[a; γ·M@a]` and `[c·a; c·γ·M@a]` are scalar multiples of each other regardless of M or γ. The measured angle is exactly 0° in feature space; any empirical deviation is PCA noise (the null floor), not coupling.

2. **The MᵀM distortion is negligible at production sparsity.** With nnz=3/50 (6% density), the random sparse M has a very small `MᵀM` eigenvalue spread; the analytic formula gives < 0.02° distortion even at coupling_scale=1. At the dense extreme the distortion is −0.28° (wrong direction for the observed cross-talk).

3. **The coupling's effective direction is shared between groups for `magnitude`.** In the production-realistic case, both groups' nuisance steps are parallel (both are `M@a` or `c·M@a`) — the coupling adds no directional difference between groups, only a scaling difference already captured by `delta`.

**What this rules out:**
- The cross-talk is not explained by the linear mean-shift coupling mechanism at any coupling strength, M structure, or dimensionality ratio.
- The mechanism is not the feature-space inner-product distortion from `MᵀM`.
- The mechanism is not amplified by nuisance block weight (dim_ratio has no effect on cross-talk, only on the null floor).

---

## Gate: Rung 5 — SNF mechanism isolation

The rung ladder has now cleared all single factors on the `concat` integration path:

| Rung | Factor | Finding |
|-----:|--------|---------|
| 0 | Linear geometry recovery | Clean; PCA k-noise floor documented |
| 1 | Methylation logit transform | Clean; M-value integration inverts the transform |
| 2 | Per-feature standardization vs projector | Standardization clean; **SNF leaks** |
| 3 | Multi-block concatenation (independent nuisance) | Clean at all dim_ratio and ρ |
| 4 | Cross-omic coupling (linear mean shift via M) | Clean; MᵀM distortion negligible |

The `concat` path is clean under all tested factors. Rung 2 identified SNF as a leaking projector. Rung 5 should isolate **which aspect of SNF** produces the cross-talk:

**Candidate Rung 5 questions:**
- A. **Affinity kernel non-linearity**: SNF uses a Gaussian affinity kernel — does the kernel's response to scale differences between groups create the angle distortion?
- B. **Graph diffusion / normalization**: The SNF diffusion step (cross-network averaging) may amplify magnitude differences into orientation differences via the graph Laplacian.
- C. **Spectral embedding**: The final PCA/spectral step applied to the fused affinity matrix may have different projection behavior than linear PCA on the feature matrix.

The minimal Rung-5 ablation is to test the SNF affinity kernel alone (no diffusion, no multi-omic fusion) on the single-block linear geometry, adding each SNF component one at a time.