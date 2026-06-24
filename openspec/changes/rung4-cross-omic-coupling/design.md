## Context

Rungs 0–3 cleared every structural factor on the `concat` path. Rung 3 confirmed that per-block z-score + concatenate + PCA is clean under independent nuisance blocks at any dimensionality ratio up to 10×: the `magnitude` angle stays at the `none` null floor, `w_anchor` tracks the naive feature-fraction prediction, and the null floor rise with `dim_ratio` is uniform k-noise dilution, not directed rotation. The specificity-study cross-talk is therefore not explained by dimensionality imbalance alone.

The one remaining untested structural factor is **cross-omic coupling**: in the production generator (`generator.py`), methylation differential indicators drive expression indicators via `incidence_cpg_gene` (a gene is differential when any of its mapped CpGs is), and expression indicators propagate to protein via `incidence_gene_protein`. This means the nuisance blocks carry *signal derived from the anchor block's step vectors* — not independent noise. The coupling introduces a second copy of the anchor geometry, projected into a different feature space via a sparse linear map M. Because M is not generally an isometry (its singular values are unequal), the combined joint-space geometry is distorted: the measured angle between groups in the PCA latent space can differ from the true anchor angle.

The distortion is analytic. With coupling matrix M (normalised so ‖M‖₂ = 1), coupling strength γ = `coupling_scale`, anchor steps `a` (group A) and `b` (group B), the joint steps after per-block z-scoring and PCA are dominated by `[a; γ·M@a]` and `[b; γ·M@b]`. Before PCA, the cosine of the joint-space angle is:

```
cos(θ_joint) = [a·b + γ²·aᵀ(MᵀM)b] / [‖[a; γ M@a]‖ · ‖[b; γ M@b]‖]
```

At γ = 0 this reduces to cos(θ_anchor). At γ > 0, the `MᵀM` term re-weights the inner product: directions in anchor space along which M has large singular values are amplified, distorting the measured angle away from the intended geometry. This is the candidate cross-talk mechanism Rung 4 tests.

## Goals / Non-Goals

**Goals:**
- A deterministic, seed-reproducible test bed that extends the Rung-3 multi-block setup by replacing the independent nuisance block with a **coupled nuisance block** whose per-group mean shift is `coupling_scale × M_norm @ anchor_step_g`, and measures `delta`/`angle` through the same per-block z-score + concatenate + PCA + estimator path.
- Characterise, per (M structure, `coupling_scale`), (a) absolute distortion of measured `delta`/`angle` vs the intended anchor-block geometry and (b) cross-talk (magnitude→`angle`, orientation→`delta`) against the Rung-3 independent-nuisance floor and a per-configuration `none` null.
- An **analytic geometry probe**: compute the closed-form joint-space angle prediction from the formula above per (M, γ, seed) and compare it to the empirical PCA-measured angle to confirm coupling geometry as the mechanism.
- Sweep `coupling_scale` (primary) and M structure (random sparse / dense / rank-1) and `dim_ratio` (secondary) to characterise onset and saturation.
- A reproducible findings writeup with the coupling-strength distortion curves, the analytic-vs-empirical comparison, and the Rung-5 gate decision.

**Non-Goals:**
- No real InterSIM incidence matrices (real p_cpg / p_gene scale requires the reference cache; deferred as a stretch goal once the synthetic mechanism is confirmed).
- No methylation→expression→protein cascade (three-block propagation); single anchor + single coupled nuisance block is the minimal one-factor addition.
- No SNF, PLS, or any projector other than per-block z-score + concatenate + PCA.
- No `evaluation.py` end-to-end harness, no RRPP rejection rates, no shape statistic.
- Not a power study.

## Decisions

**1. Nuisance block mean is `coupling_scale × M_norm @ anchor_step_g`; noise is unchanged.**
The anchor block is generated identically to Rung 3 (step construction via `linear_recovery.generate_dataset`, then fresh RNG for sample noise). The nuisance block's per-cell mean is `coupling_scale × M_norm @ step_g_stage`, where `step_g_stage` is zero at stage 0 and the group's step vector at stage 1. Independent Gaussian noise with the same `noise_scale` σ is added to every nuisance sample, so the nuisance block is `N(μ_nuis, σ²I)` with `μ_nuis = coupling_scale × M_norm @ step`. This is the minimal coupling: one new free parameter (`coupling_scale`) and one new structural object (M), everything else from Rung 3 unchanged. Alternative considered: propagate the coupling through the indicators (binary support from anchor → binary support for nuisance) as in the real generator — rejected because it conflates the coupling geometry with the indicator sampling distribution, breaking the one-factor discipline; continuous coupling via M is the pure structural test.

**2. M is operator-norm normalised: `M_norm = M / σ_max(M)`.**
`σ_max(M)` is the largest singular value of M, computed via full SVD (p_anchor ≤ 50 in all sweeps; trivially fast). After normalisation, ‖M_norm‖₂ = 1, so `coupling_scale × M_norm @ a_hat` has norm ≤ `coupling_scale × ‖a‖` (with equality when `a` aligns with M's dominant right singular vector). This makes `coupling_scale` directly interpretable as the maximum fraction of the anchor step magnitude that the coupling can inject into the nuisance block, regardless of M's shape or sparsity. Alternative: Frobenius normalisation — rejected because it depends on the number of non-zeros in M, conflating coupling geometry with matrix density.

**3. Three M structures probe different distortion regimes.**
The analytic formula shows distortion is driven by the eigenvalue spectrum of `MᵀM`. Three constructions target the key regimes:
- **Random sparse** (`nnz_per_nuis = 3`): each nuisance feature independently links to 3 anchor features drawn uniformly at random (expected density `3 / p_anchor = 6%` at p_anchor = 50). This is the production-realistic case; `MᵀM` has a broad spectrum with no dominant structure.
- **Dense** (all-ones matrix): `M = 1^{p_nuis × p_anchor}` — all nuisance features link to all anchor features. After operator-norm normalisation, `M_norm @ a = (Σ a_j / (√p_nuis × √p_anchor)) × 1^{p_nuis}` — a constant vector whose direction is fixed regardless of `a`. The coupling adds the same nuisance step direction for every group, so at the joint level both groups' nuisance components are identical (the direction of their nuisance step is constant). This is a maximum-density extreme; `MᵀM` has a single dominant eigenvalue aligned with the all-ones direction, so the distortion is maximal for anchor steps that have a non-zero sum and zero for steps in the null space of the all-ones direction.
- **Rank-1** (`M = u vᵀ` where `u = 1^{p_nuis}`, `v = e_k` for a fixed anchor feature index k): all nuisance features link to a single anchor feature. After normalisation, `M_norm @ a = a_k × 1^{p_nuis}` — the coupling projects only one component of the anchor step. `MᵀM = e_k e_kᵀ` (rank-1, eigenvalue 1 in direction `e_k`). This isolates the "one feature drives everything" extreme.

The random-sparse structure is the headline; dense and rank-1 bracket the distortion range.

**4. `coupling_scale` ∈ {0, 0.25, 0.5, 0.75, 1.0} is the primary swept variable.**
`coupling_scale = 0` is the Rung-3 independent-nuisance baseline (the nuisance block has zero mean, identical to Rung 3's `rho_nuisance = 0` arm). `coupling_scale = 1` is the maximum coupling. The sweep locates the onset and saturation of any distortion. M is re-seeded independently of the anchor geometry seed so the structural variability of M is averaged separately (see Decision 5).

**5. M is seeded separately from the anchor geometry; both are averaged over the headline seed set.**
The anchor geometry seed (0–9) controls the step vectors and anchor noise. The coupling matrix M is constructed with a dedicated `matrix_seed` parameter. The headline run fixes `matrix_seed = 0` (one fixed M per M-structure type) and averages `delta`/`angle` over anchor seeds 0–9. A secondary probe averages over `matrix_seeds` 0–4 at the headline `coupling_scale = 0.75` to confirm results are not matrix-seed-specific.

**6. Analytic prediction is computed in joint feature space (pre-PCA).**
The formula `cos(θ_joint) = [a·b + γ²·aᵀ(MᵀM)b] / [‖[a; γ M@a]‖ · ‖[b; γ M@b]‖]` uses the true anchor step vectors (`step_A`, `step_B`) and the normalised M directly — no fitting required. The predicted angle `θ_pred` is computed per (anchor seed, coupling_scale) and reported alongside the empirical PCA angle `θ_meas`. The ratio `θ_pred / θ_anchor` (where `θ_anchor = 45°`) measures the feature-space distortion attributable purely to the coupling; `θ_meas / θ_pred` measures the residual PCA projection effect. If the coupling formula accounts for most of the distortion above the Rung-3 baseline, the mechanism is confirmed. Alternative: simulate a ground-truth PCA recovery on the coupled population means (no noise) and compare to the formula — included as a secondary diagnostic to confirm the formula applies before PCA.

**7. Magnitude as a within-run no-cross-talk control (analytic proof).**
For `manipulation = "magnitude"`, `b = c × a`, and the analytic formula gives:
```
cos(θ_joint) = [a·(c·a) + γ²·aᵀ(MᵀM)(c·a)] / [‖[a; γ M@a]‖ · ‖[c·a; c·γ M@a]‖]
             = c × (‖a‖² + γ²·‖M@a‖²) / (‖[a; γ M@a]‖ · c · ‖[a; γ M@a]‖)
             = 1
```
So `θ_joint = 0°` analytically for the magnitude manipulation — the coupling preserves direction because both groups' nuisance steps are scalar multiples of the same vector. The empirical `magnitude` arm angle should therefore stay at the `none` null floor at every `coupling_scale`, providing a positive control. Any deviation is attributable to PCA noise (the null floor), not to coupling cross-talk.

**8. `build_joint_matrix` from `multiblock_recovery` is reused without modification.**
The coupling test bed produces a dataset with `X_anchor` and `X_nuisance` attributes (same interface as `MultiblockRecoveryDataset`). `build_joint_matrix` from `simulations/multiblock_recovery.py` applies the per-block z-score + concatenate transform and is imported directly — no reimplementation. The estimator path is identical to Rungs 0–3.

**9. Dim_ratio secondary sweep at {0.5, 1, 5}.**
At higher `dim_ratio`, the nuisance block contributes more columns to the joint matrix, increasing the weight of the coupling signal in the PCA. From the analytic formula, the coupling term `γ²·MᵀM` is present regardless of `dim_ratio`, but the weight of the nuisance block in the joint PCA increases with `dim_ratio`, amplifying the coupling's influence on the measured angle. The secondary sweep at `dim_ratio ∈ {0.5, 1, 5}` (fixed `coupling_scale = 0.75`) checks whether the distortion scales with the nuisance block weight, which would confirm that the amplification mechanism is the joint PCA — not a feature-space effect independent of projection.

## Risks / Trade-offs

- [The analytic formula ignores PCA projection and per-block standardisation] → The formula is a feature-space prediction, not a latent-space prediction; PCA introduces additional distortion (the k-noise floor, documented in Rung 0). The comparison ratio `θ_meas / θ_pred` separates coupling-geometry distortion from PCA-projection distortion; if `θ_pred` explains the bulk of the lift above the baseline, the coupling mechanism is confirmed even if the residual `θ_meas / θ_pred` is non-trivial.
- [Per-block z-scoring changes the effective coupling strength] → After z-scoring, each block is standardised to unit-variance features, which rescales `M_norm @ step` relative to the noise floor. The analytic formula is computed with the pre-standardisation step vectors; the coupling is applied before standardisation (in the mean shift), so standardisation then re-scales both signal and noise. A secondary "formula-after-standardisation" check verifies the prediction holds with per-block-standardised steps.
- [Dense M conflates all anchor directions into one nuisance direction] → This is by construction; dense M is a structural extreme (upper bound), not a production model. Label it explicitly as an extreme and do not over-interpret its distortion as representative of the real incidence structure.
- [Rank-1 M picks an arbitrary anchor feature] → Fix `k = 0` (the first feature) so the rank-1 M is deterministic and reproducible; note that the distortion depends on how much of the anchor step projects onto `e_0`, which varies with seed. Average over anchor seeds 0–9 as usual.
- [SVD of M at large dim_ratio / large p_nuis] → At `dim_ratio = 5, p_nuis = 250`: SVD of a 250 × 50 matrix is instantaneous. No computational risk across any tested configuration.
- [MC noise at small coupling_scale may mask onset] → The `none` null at each `coupling_scale` serves as the per-configuration floor; onset is visible as the `orientation` arm lifting above the `none` floor, not as absolute angle change. The 10-seed average gives stable floor estimates.

## Migration Plan

Not applicable — additive only. One new module `src/motco/simulations/coupling_recovery.py`, its unit tests `tests/test_coupling_recovery.py`, and a driver script `scripts/coupling_recovery_probe.py`. No existing module, spec, or public API changes.

## Open Questions

- Whether to add a **no-noise population-mean check** (draw zero noise, compute the exact joint angle between population mean trajectories) as a zero-variance analytic sanity check before running the full stochastic sweep.
- Whether the secondary probe averaging over `matrix_seeds` should be folded into the headline table or reported as a separate stability check.
- Whether the Rung-5 gate should point at the **real InterSIM incidence matrices** (same coupling mechanism, production dimensionalities) or at the **full end-to-end `evaluation.py` path** (which adds RRPP, the semisynthetic generator, and the three-block cascade) — depending on whether the synthetic coupling at realistic `coupling_scale` reproduces the specificity-study cross-talk magnitude.