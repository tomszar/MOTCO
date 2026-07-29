## Context

Rungs 0–2 cleared single-block factors one by one. Rung 0 proved the linear floor is clean under mean-centered PCA. Rung 1 showed the methylation `rev.logit` is inverted by M-value integration and is not a standing cross-talk source. Rung 2 showed that per-feature standardization — the transform inside the production `concat` projector — is also clean on a single block: it preserves orientation, introduces no magnitude→orientation cross-talk, and is robust to feature-scale heterogeneity. The specificity study's cross-talk is therefore not explained by any single-block factor.

The Rung-2 gate decision names **heterogeneous multi-omic concatenation** as the next single untested factor. In the production `concat` pipeline, each omic block is z-scored independently and then concatenated into a joint feature matrix before PCA. When blocks differ in dimensionality or correlation structure, the per-block variance contributions to the pooled matrix are not equal — the larger or more correlated block dominates the top principal components, potentially rotating them away from the geometry-carrying (anchor) block's subspace. That rotation can express as magnitude→orientation cross-talk even when each block in isolation is clean.

Rung 3 adds exactly that one factor — multiple heterogeneous blocks — and nothing else. The geometry injection, estimator path, seed set, and reporting conventions are identical to Rungs 0–2.

## Goals / Non-Goals

**Goals:**
- A deterministic, seed-reproducible test bed that injects the known Rung-0 2-stage geometry into a single **anchor block** (M-value space, no `rev.logit`) and appends one or two independent **nuisance blocks** of configurable dimensionality and exchangeable-correlation structure, applies per-block z-score + concatenate + PCA (the production `concat` transform), and measures `delta`/`angle` via the existing `stats/trajectory.py` estimators.
- Characterise, per block configuration, (a) absolute distortion of measured `delta`/`angle` vs the intended anchor-block geometry and (b) cross-talk (magnitude→`angle`, orientation→`delta`) against the single-block PCA floor and a per-configuration `none` null.
- Sweep **nuisance-block dimensionality ratio** (`p_nuisance / p_anchor`) and **anchor effect size** (`signal_scale`) to locate any onset and determine whether it scales with block-size imbalance.
- A **block-weight decomposition** probe: quantify, from the joint PCA loadings, the fraction of top-k explained variance attributable to the anchor block vs the nuisance block(s) as a function of the dimensionality ratio, to determine whether cross-talk onset coincides with the variance-fraction tipping point.
- A reproducible findings writeup with the per-configuration distortion/cross-talk table, the block-weight curve, and the Rung-4 gate decision.

**Non-Goals:**
- No cross-omic coupling: the nuisance blocks are independent pure noise; signal correlation between blocks (the InterSIM incidence-map mechanism) is the Rung-4 candidate.
- No real per-block dimensionalities from the generator reference; the dimensionality ratio is a synthetic sweep.
- No SNF multi-block fusion (recorded as a known-leaky projector in Rung 2; no further analysis here).
- No PLS latent-space integration over multi-block inputs.
- No `evaluation.py` end-to-end harness, no RRPP rejection rates, no Procrustes shape statistic, no purity metric.
- Not a power study — this is a mechanistic block-imbalance characterisation.

## Decisions

**1. Anchor block carries the geometry; nuisance blocks are independent pure noise.**
The anchor block is generated using the same machinery as Rung 0 (`linear_recovery.generate_dataset`, which produces an `(n_samples × p_anchor)` matrix with known per-(group, stage) step vectors in a clean M-value frame). Each nuisance block is drawn from a zero-mean MVN with exchangeable covariance `Σ_ν = σ²[(1−ρ)I + ρ11ᵀ]`, independently of the anchor and of other nuisance blocks. Nuisance blocks carry no signal by construction — any cross-talk that appears is purely a consequence of the multi-block projection, not of signal leakage between blocks. Alternative considered: inject a proportional subthreshold signal into nuisance blocks — rejected because it would conflate block-imbalance geometry distortion with cross-block signal amplification, breaking the one-factor discipline.

**2. Per-block z-score + concatenate + PCA is the exact production `concat` transform.**
Each block is z-scored independently (feature-wise; zero-variance columns replaced with zeros to match `evaluation.py:_concat_integration`), then the standardised blocks are column-concatenated into a joint `(n_samples × p_total)` matrix where `p_total = p_anchor + n_nuisance × p_nuisance`. PCA with `n_components` retained components is then fit on this joint matrix. This is exactly what `evaluation.py:_concat_integration` does; no reimplementation is needed — the test bed calls the same NumPy / scikit-learn operations in the same order. The estimator path (`get_model_matrix`/`build_ls_means`/`estimate_difference`, 2-group × 2-stage, two-group contrast) is identical to Rungs 0–2.

**3. Nuisance-block dimensionality ratio is the primary swept variable; nuisance-block correlation is secondary.**
`dim_ratio = p_nuisance / p_anchor` is swept over `{0.5, 1, 2, 5, 10}`, representing a range from nuisance-smaller-than-anchor to severely nuisance-dominated (10× more nuisance features than anchor features per block, approximating the real CpG-vs-gene imbalance). At `dim_ratio = 0` (or `n_nuisance_blocks = 0`) the test bed reproduces the Rung-2 single-block `standardize` floor exactly. Nuisance-block exchangeable correlation `ρ_nuisance` ∈ {0, 0.3, 0.7} is a secondary sweep: correlated features in the nuisance block increase its effective rank contribution to the pooled covariance, which may amplify or suppress the block-imbalance effect differently from dimensionality alone.

**4. Block-weight decomposition from PCA loadings.**
After fitting PCA on the joint matrix, the fraction of variance in the top-k components attributable to the anchor block is `w_anchor(k) = ‖V_anchor‖²_F / ‖V‖²_F`, where `V_anchor` is the sub-matrix of the loading matrix `V ∈ R^{p_total × k}` restricted to the anchor-block rows, and norms are Frobenius. A `w_anchor` close to 1 means the top components align with the anchor block; as `dim_ratio` grows, `w_anchor` shrinks toward `p_anchor / p_total` (the naive feature-count fraction). If cross-talk onset coincides with `w_anchor` falling below a threshold (e.g. < 0.5), that is evidence that variance-fraction dominance — not any other mechanism — drives it. This decomposition is computed in a dedicated `decompose_block_weight` function and plotted as a curve vs `dim_ratio`.

**5. Nuisance-block noise scale is matched to the anchor.**
Nuisance blocks use the same per-feature noise scale `σ` as the anchor block (`noise_scale` parameter). Per-block z-scoring then standardises both blocks to unit variance per feature before concatenation, which is exactly the production behaviour. Intentionally mismatching scales would be a redundant sweep (Rung 2 already showed per-feature standardisation is robust to scale heterogeneity); keeping them equal isolates the dimensionality/correlation effect cleanly.

**6. `n_nuisance_blocks` ∈ {0, 1, 2}; results are reported per `(n_nuisance_blocks, dim_ratio, ρ_nuisance)` cell.**
Zero nuisance blocks is the single-block baseline; one adds a single nuisance block; two adds two independently drawn nuisance blocks (both with the same `dim_ratio` and `ρ_nuisance`). The headline table uses `n_nuisance_blocks = 1`, which is the minimal departure from the Rung-2 floor; the two-block arm checks whether effects compound additively.

**7. Measurement, seeds, and reporting are identical to Rungs 0–2.**
Seeds 0–9; manipulations `none`/`magnitude`/`orientation`; `n_components = 10` headline (sweep 2/5/10/20 secondary); `signal_scale = 5.0`, `scale_c = 2.0`, `angle_theta = 45°`. Cross-talk is read against the same-configuration `none` null and against the `dim_ratio = 0` (single-block) floor.

## Risks / Trade-offs

- [PCA may still recover anchor geometry if `p_anchor` is enough to dominate despite small `dim_ratio`] → The dimensionality sweep goes to 10×; at that ratio the nuisance block contributes 90 % of features and is expected to dominate the pooled covariance; if even this does not produce cross-talk, the result is informative and the gate points to Rung 4 (cross-omic coupling).
- [Block-weight decomposition conflates direction of loading with magnitude] → The Frobenius norm of the loading sub-matrix is direction-agnostic; a low `w_anchor` means anchor features are weakly represented in the top-k components regardless of the sign or direction of individual loadings, which is the relevant quantity for distortion.
- [Independent nuisance blocks may not stress the PCA enough at small `dim_ratio` because noise PC is orthogonal to signal PC] → At small ratio the nuisance block is narrow and its top noise PCs are absorbed into components beyond k; as ratio grows the nuisance eigenvalues rise relative to the anchor signal eigenvalues and eventually crowd out the signal. This is exactly the mechanism being tested.
- [MC noise confounds onset detection] → Averaging over seeds 0–9 and reading against the per-configuration `none` floor suppresses seed noise, so onset is visible as a lift in the `magnitude`→`angle` cross-talk curve above the `none` floor, not as scatter.
- [Two independent nuisance blocks at high `dim_ratio` produce a very high-dimensional joint matrix] → At `p_anchor = 50`, `dim_ratio = 10`, `n_nuisance_blocks = 2`: `p_total = 50 + 2 × 500 = 1050`. PCA on 160 samples × 1050 features is fast (thin SVD); no computational risk.

## Migration Plan

Not applicable — additive only. One new module `src/motco/simulations/multiblock_recovery.py`, its unit tests `tests/test_multiblock_recovery.py`, and a driver script `scripts/multiblock_recovery_probe.py`. No existing module, spec, or public API changes. The change is archived once findings land, following the Rung-0/1/2 pattern.

## Open Questions

- Whether to include a **block-specific signal-recovery ratio** (angle between the true anchor step projected into the joint PCA vs the full-feature-space step direction) as a more direct measure of geometric distortion than the estimator-level `delta`/`angle`, or whether estimator-level reporting is sufficient and more interpretable.
- The appropriate `n_components` for the headline table: at high `dim_ratio` the signal may fall outside the top `k` components; a secondary sweep over `k` relative to `dim_ratio` may be needed to disentangle "geometry lost from retained components" from "geometry rotated into non-signal direction."
- Whether the findings writeup should include a **closed-form prediction** for `w_anchor(k)` as a function of `dim_ratio` and `ρ_nuisance` (derivable from Marchenko–Pastur eigenvalue scaling for isotropic nuisance) as a theoretical sanity-check on the numerics.
- Whether the Rung-4 gate points at **cross-omic coupling** (correlated signal between the anchor and nuisance blocks via the InterSIM incidence maps) or at a different single factor, depending on whether this rung finds cross-talk or not.