## Context

Rung 0 proved the linear floor is clean under a single, deliberately benign projector: mean-centered PCA with no per-feature standardization. Rung 1 proved the methylation `rev.logit` is invertible by M-value integration and is not a standing cross-talk source, and — because `logit` acts coordinate-wise — mooted the heterogeneous-baseline rung. The specificity study's magnitude→orientation cross-talk therefore enters above the methylation representation. The obvious next single factor is the **projector**: the production trajectory pipeline does not use Rung 0's PCA — `evaluation.py` standardizes-and-concatenates (`concat`) or builds an SNF spectral embedding (`snf`), and the package also ships a supervised PLS-DA projector (`stats/pls.py`). Each of these transforms the clean injected geometry before `estimate_difference` measures it, and a standardizing, label-conditioned, or graph-spectral map can rotate or rescale a similarity transform that PCA leaves intact.

Rung 2 holds everything from Rung 0/1 fixed (geometry injection, two stages, M-value space so there is no `rev.logit` distortion, the production estimators, the seed-averaging and null-floor reporting) and varies **only the projector**. It reuses the Rung-0/1 helpers and the production `stats/pls.fit_plsda_transform` and `stats/snf` functions unchanged, so the projectors under test are the package's own, not reimplementations.

## Goals / Non-Goals

**Goals:**
- A deterministic, seed-reproducible test bed that injects the known Rung-0 2-stage geometry in a clean M-value feature space and measures `delta`/`angle` through a **selectable projector**.
- Compare four projectors on identical data: **PCA** (reference floor), **standardize** (the `concat` transform), **PLS-DA** (supervised), **SNF** spectral (graph-spectral).
- Report, per projector, absolute distortion vs the intended geometry and cross-talk (magnitude→`angle`, orientation→`delta`) against the PCA floor and a per-projector `none` null; sweep effect size and latent dimensionality to find any onset.
- A dedicated **supervised-leakage probe**: detect whether PLS-DA aligns a `none` (null) trajectory with the group axis.
- A reproducible findings writeup with the per-projector table and the Rung-3 gate decision.

**Non-Goals:**
- No true multi-omic concatenation of heterogeneous-scale blocks (single block here; heterogeneous concat is a candidate Rung 3).
- No cross-omic cascade, full InterSIM generator, or `rev.logit` distortion (methylation is fixed in M-space — the clean linear frame).
- No `evaluation.py` end-to-end harness, no RRPP rejection-rate sweep, no shape statistic, no purity metric.
- Not a power study — this is a mechanistic projector-distortion characterization.

## Decisions

**1. Reuse the Rung-0/1 geometry injection in M-value space; vary only the projector.**
Per-(group, stage) means are built exactly as in Rung 0 (`a_feat`, `step_B` for `none`/`magnitude`/`orientation`), drawn with isotropic MVN noise. Because Rung 1 established M-value integration as the correct representation, the feature space *is* M-space — there is no `rev.logit` step, so the only thing distinguishing Rung 2 from the Rung-0 clean floor is the projector. Alternative considered: layer the projector swap on top of β/`rev.logit` — rejected, because that would conflate the (already-characterized) nonlinearity with the projector and break the one-factor discipline.

**2. A single `projector` knob with four options; a uniform measurement contract.**
Each projector maps the `(n_samples × n_features)` feature matrix to a `(n_samples × n_components)` latent matrix `Y`, after which the *identical* Rung-0 measurement runs: 2-group × 2-stage design via `get_model_matrix`/`build_ls_means`, two-group contrast, `estimate_difference` → `delta`/`angle` (shape `nan`, dropped). The projectors:
- `pca`: `PCA(n_components)` on mean-centered features (Rung-0 reference).
- `standardize`: per-feature z-score (matching `evaluation.py:_concat_integration`, `std==0 → 1`), then `PCA(n_components)` — isolating standardization from the PCA rotation by composing them so the difference from `pca` is *only* the scaling.
- `plsda`: `fit_plsda_transform(X, y, n_components)` with the supervised label `y` (see Decision 3).
- `snf`: `get_affinity_matrix` per (single) block → `SNF` → `get_spectral(n_components)`.
Alternative considered: route through `evaluation.integrate_semisynthetic_dataset` — rejected because that harness expects a multi-omic `SemiSyntheticTrajectoryDataset` and bundles RRPP; calling the projector functions directly keeps the rung single-factor and fast.

**3. PLS-DA label is the group; the `none` null is the leakage control.**
PLS-DA is supervised and needs a class label. The trajectory question is about *group* differences, so the label is the group membership (A/B), matching how a practitioner would use PLS-DA to separate groups before inspecting trajectories. This is exactly the configuration that can manufacture group-aligned latent structure, so the `none` manipulation (identical group trajectories) is the critical control: under an honest projector `none` stays at the null floor; under label leakage PLS-DA invents a group separation. Alternative labels considered: stage, or group×stage — noted as a secondary probe, but group is the headline because it is the production classification target and the worst-case leakage source.

**4. Cross-talk is read against two references: the PCA floor and the per-projector `none` null.**
Absolute distortion compares measured `delta`/`angle` to the intended targets (`signal_scale·(c−1)`, `θ`). Cross-talk (magnitude→`angle`, orientation→`delta`) is read against the *same projector's* `none` floor at the same settings, so a projector that is merely noisy is separated from one that systematically rotates/rescales. The PCA floor is the cross-projector baseline: a clean projector matches PCA within tolerance. Reporting is mean ± spread over the Rung-0 seed set.

**5. Latent dimensionality is swept per projector.**
PCA/standardize/PLS-DA take `n_components`; SNF takes `spectral_components`. The injected geometry lives in a low-dimensional subspace, but a projector may need enough components to retain it (or, for SNF, may distort it regardless of `k`). A small sweep over component count locates whether cross-talk is a dimensionality artifact (too few retained directions) or intrinsic to the projector. Effect size (`signal_scale`/`c`/`θ`) is swept jointly to find any onset, as in Rung 1.

**6. SNF angles are interpreted with care.**
SNF spectral coordinates are eigenvectors of a fused sample-similarity graph: they have no fixed sign, scale, or feature-space alignment, so a measured `angle`/`delta` in SNF latent space is not directly comparable in *units* to PCA. The test bed therefore reports SNF results primarily as *recovery vs null* (does the injected geometry separate from `none`?) and flags that absolute magnitude is embedding-relative, rather than asserting a unit-matched distortion. This is itself a finding: a projector whose latent metric is not commensurable with the injected geometry is a cross-talk risk for the production `snf` path.

**7. Measurement, seeds, and reporting are identical to Rung 0/1.**
No per-feature standardization is applied *except* inside the `standardize` projector (that is the point of that arm); the `pca` arm stays mean-centered-only so it reproduces the Rung-0 floor exactly. Same seed set, same "near zero relative to the `none` floor and a stated tolerance" language.

## Risks / Trade-offs

- [A clean result is near-tautological if the geometry is forced into every projector's retained subspace] → Sweep component count and effect size and report recovery as a function of them, so "clean" is conditional and informative, not constructed.
- [PLS-DA leakage could be masked if `n_components` is large relative to samples (overfitting separates everything)] → Probe `none` explicitly across a component sweep and small sample sizes; report the `none`-floor inflation directly rather than only the signal arms.
- [SNF latent metric is not unit-commensurable with feature-space geometry] → Report SNF as recovery-vs-null and label absolute magnitudes embedding-relative (Decision 6); do not over-claim a numeric distortion factor.
- [Standardize arm is degenerate on isotropic noise] → With isotropic MVN noise all features share scale, so z-scoring is near-identity and the `standardize` arm may match `pca` trivially. Mitigation: optionally inject heteroscedastic per-feature noise (a single anisotropy knob) so standardization actually does something, while keeping the geometry known; flagged as an open question on whether to include heteroscedasticity in the headline run.
- [Cross-talk confounded by MC noise] → Average over the Rung-0 seed set and read cross-talk against the per-projector `none` floor at the same settings.

## Migration Plan

Not applicable — this is additive: one new module under `src/motco/simulations/`, its tests, and a driver script. No existing module, spec, or public API changes. The change is archived once findings land, following the Rung-0/1 pattern.

## Open Questions

- Whether to include a **heteroscedastic-noise** knob so the `standardize` arm is non-trivial in the headline run, or keep isotropic noise and treat standardization-on-isotropic as a (informative) near-identity baseline with anisotropy as a secondary sweep.
- The PLS-DA **component count** for the headline table, and whether to add a small-sample leakage stress point where overfitting is most likely.
- Whether SNF's `K`/`k`/`t` defaults from `evaluation.py:_snf_integration` are reused verbatim or pinned to test-bed-appropriate values given the small synthetic sample sizes.
- Whether the findings writeup stays in the change folder (matching Rung 0/1) or graduates to `src/motco/simulations/` for longer-term visibility.
- Whether the Rung-3 gate points at heterogeneous multi-omic **concatenation** or the **cross-omic coupling**, decided by which projector(s) leak here.
