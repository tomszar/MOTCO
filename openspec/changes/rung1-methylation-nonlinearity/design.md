## Context

Rung 0 proved the linear floor is clean: PCA + the production estimators isolate `delta` and `angle` with no cross-talk when the geometry is injected directly in the measurement frame. The specificity study's cross-talk therefore arises above Rung 0. The Rung-0 findings finger the methylation `rev.logit` as the most likely cause: in InterSIM, the trajectory is a clean similarity transform in M-value (logit) space, but methylation is *measured* as β = `rev_logit(M)`, a saturating nonlinearity. A straight step in M-space becomes a compressed, curved step in β-space, and the amount of distortion depends entirely on where on the sigmoid the step sits.

Rung 1 adds this one factor. It reuses the Rung-0 generation and inverse-design helpers (`simulations/linear_recovery.py`) and the production estimators (`stats/trajectory.py`) unchanged, and reuses `generator.rev_logit` so the nonlinearity is the package's, not a reimplementation.

## Goals / Non-Goals

**Goals:**
- A deterministic, seed-reproducible methylation test bed: inject the known Rung-0 2-stage geometry in M-value space, push through `rev_logit`, project with inline PCA, measure `delta`/`angle`.
- Make the sigmoid **operating point** an explicit parameter and sweep it from center (≈ linear) to tails (saturating); report `delta`/`angle` recovery and cross-talk versus operating point.
- Quantify both absolute distortion (β-space measured vs M-space intended) and cross-talk (magnitude→`angle`, orientation→`delta`) against the Rung-0 null floor.
- A reproducible findings writeup with the distortion curves and the Rung-2 gate decision.

**Non-Goals:**
- No expression/protein layer, cross-omic cascade, or full InterSIM generator (the cascade is a later rung).
- No PLS projector, no shape statistic, no per-feature standardization, no purity metric.
- No real per-CpG `mean_M` baselines — the sweep uses synthetic, controllable operating points (the realistic anchor is deferred).
- Not a power study — this is a mechanistic distortion characterization, not a rejection-rate sweep.

## Decisions

**1. Methylation-only injection; gene/protein excluded.**
Only methylation carries the injected geometry, and only methylation is measured. Rationale: InterSIM's downstream coupling (`generator.py:174-186`) blends a *constant* `methyl_gene_level_mean` into the gene mean and transmits trajectory structure to gene/protein only through the shared differential *indicator support* — the downstream shift *magnitude* is governed by an independent `delta_expr`, decoupled from the methylation shift. A "let methylation drive gene drive protein" path would therefore add the indicator-coupling and mean-blend factors on top of `rev.logit`, conflating three changes. Rung 1 keeps `rev.logit` as the single new variable versus Rung 0. (Decision confirmed with the maintainer.)

**2. Reinterpret the Rung-0 feature matrix as M-values; reuse the generation path.**
Per-(group, stage) M-space means: `μ_{A,0} = m_baseline`, `μ_{A,1} = m_baseline + a_feat`, `μ_{B,0} = m_baseline`, `μ_{B,1} = m_baseline + step_B`, where `a_feat` and `step_B` (none/magnitude/orientation transforms) are constructed exactly as in Rung 0. Samples are drawn `x_M = μ + N(0, σ²I)` (isotropic, as Rung 0) and then `β = rev_logit(x_M)`. The Rung-0 `generate_dataset` step-construction logic is reused; only the baseline offset and the trailing `rev_logit` are new. Alternative considered: a fresh generator — rejected to keep "everything else identical to Rung 0" literally true.

**3. Operating point `m_baseline` is the independent variable, swept center→tail.**
`rev_logit` has slope `β(1−β)`: ≈ 0.25 at M = 0 (β = 0.5) and → 0 as |M| grows. At the center a step passes through nearly affinely (Rung 1 ≈ Rung 0 floor); on the tails it is compressed and bent. The headline result is `delta`/`angle` recovery and cross-talk as a function of `m_baseline` (scalar offset applied to all CpGs for a clean, interpretable mechanism). Without this sweep the experiment is null by construction, so the sweep *is* the experiment.

**4. Step scale is a secondary lever.**
Even at the center, a *large* step spans a wide arc of the sigmoid and so curves. A secondary sweep over `signal_scale` (with `m_baseline` fixed at center) shows the distortion that comes from step span rather than baseline offset, separating the two routes to nonlinearity.

**5. Ground-truth references are dual.**
Measured β-space `delta`/`angle` are compared against (a) the *intended M-space* geometry (`delta` target = `signal_scale·(c−1)`, `angle` target = `θ`) to quantify absolute distortion, and (b) the *Rung-0 linear floor* to confirm any cross-talk is new. The `none` control calibrates the null floor at each operating point, exactly as in Rung 0.

**6. Inverse design is first-order (Jacobian) and secondary.**
The nonlinearity breaks Rung 0's exact PCA round-trip. To keep the feature-intuition thread alive, the inverse design is linearized at the operating point: the local map M→β is `J = diag(β(1−β))`, so a target β-space latent step pulls back to an M-space change via the Rung-0 formula composed with `J⁻¹`. This is explicitly first-order (accurate near the operating point, degrading into saturation) and is reported as a secondary analysis, not the headline.

**7. Measurement, projection, seeds, and reporting are identical to Rung 0.**
Inline PCA on mean-centered β (no per-feature standardization), 2-group × 2-stage design via `get_model_matrix`/`build_ls_means`/contrast, `estimate_difference` → `delta`/`angle` (shape `nan`, dropped), mean ± spread over the same seed set, "near zero" stated relative to the `none` floor and a tolerance.

## Risks / Trade-offs

- [Sweep could show "no distortion" if the chosen operating-point range stays near the linear center] → Sweep out to genuine saturation (e.g. β baselines approaching 0.9/0.1 and beyond) and report the slope `β(1−β)` alongside, so the range demonstrably covers the nonlinear regime.
- [Scalar `m_baseline` is less realistic than per-CpG `mean_M`] → Intended: a uniform operating point gives a clean, interpretable mechanism; the realistic per-CpG anchor is explicitly deferred so the curve is read first, the realistic point placed on it later.
- [First-order inverse design misleads in saturation] → Label it first-order, report it only as a continuity analysis, and note where on the sweep the linearization error becomes large.
- [Cross-talk could be confounded by MC noise rather than the nonlinearity] → Average over the Rung-0 seed set and read cross-talk against the `none` floor at the *same* operating point, so the nonlinearity is the only thing varying.

## Open Questions

- The exact `m_baseline` grid and the headline operating point for the summary table — pick a range that spans clearly-linear to clearly-saturated, then report the cross-talk onset.
- Whether to retain Rung 0's k = 3..10 component sweep here or fix `k` and spend the budget on the operating-point grid (leaning: fix `k` at the Rung-0 default, note `k` is orthogonal to the nonlinearity).
- Whether the findings writeup stays in the change folder (matching Rung 0) or graduates to `src/motco/simulations/` for longer-term visibility.
