## 1. Test-bed module scaffold

- [ ] 1.1 Create `src/motco/simulations/methylation_recovery.py` with a frozen params dataclass extending the Rung-0 config with `m_baseline` (scalar M-value operating-point offset) and reusing the Rung-0 fields (seed, n_features, per-cell sample size, noise/signal scale, manipulation `none`/`magnitude`/`orientation`, scale `c`, angle `θ`, PCA `n_components`), plus a dataset dataclass (β matrix + group/stage labels + the known M-space step vectors and the M-space baseline).
- [ ] 1.2 Implement generation by reusing the Rung-0 step construction (`a_feat`, `step_B` per manipulation) for the M-space means `μ_{g,s} = m_baseline (+ step_g at stage 1)`, drawing `x_M = μ + N(0, σ²I)` from a seeded `np.random.default_rng`, and applying `generator.rev_logit` to obtain β. Reuse Rung-0 helpers rather than copying them.
- [ ] 1.3 Validate params (≥2 samples per cell, `n_components` in range, positive noise/signal scales, finite `m_baseline`) and raise a clear module-specific error on bad input.

## 2. Linear projection and measurement

- [ ] 2.1 Fit inline PCA (mean-centered β, no per-feature standardization) on the pooled β matrix; retain `n_components`; expose loadings `Vₖ`.
- [ ] 2.2 Project β to latent coordinates and build the 2-group × 2-stage design via `get_model_matrix` / `build_ls_means` and the two-group contrast (identical to Rung 0).
- [ ] 2.3 Call `estimate_difference` on the projected `Y`; extract `delta` and `angle` for the group pair (shape is `nan` at 2 stages and dropped).

## 3. Operating-point and step-scale sweeps

- [ ] 3.1 Add an operating-point driver that sweeps `m_baseline` from sigmoid center to tails for each manipulation over the Rung-0 seed set, returning mean ± spread of `delta`/`angle` per (manipulation, operating point), with `none` as the per-point null floor and the sigmoid slope `β(1−β)` recorded alongside.
- [ ] 3.2 Add a secondary `signal_scale` sweep at fixed center `m_baseline` to separate step-span distortion from baseline-offset distortion.
- [ ] 3.3 Compute distortion metrics: absolute (measured β-space `delta`/`angle` vs intended M-space `signal_scale·(c−1)` / `θ`) and cross-talk (magnitude→`angle`, orientation→`delta`) read against the `none` floor at the same operating point.

## 4. First-order inverse design (secondary)

- [ ] 4.1 Implement a Jacobian-linearized preimage: build `J = diag(β(1−β))` at the operating point and pull a target β-space latent step back to an M-space change via the Rung-0 inverse-design formula composed with `J⁻¹`; assert the first-order round-trip holds near the center and document where saturation breaks it.
- [ ] 4.2 Reuse the Rung-0 `delta_x_summary` to report support/sparsity of the linearized M-space recipes for magnitude vs orientation, noting the first-order caveat.

## 5. Tests

- [ ] 5.1 Unit test: deterministic output for a fixed seed; exactly two stages; β ∈ (0,1); per-cell β means approximate `rev_logit` of the M-space spec.
- [ ] 5.2 Unit test (center ≈ Rung-0 floor): at `m_baseline` = 0 with a small step, `magnitude` moves `delta` and `orientation` moves `angle` with cross-talk near the `none` baseline — i.e. Rung 1 reduces to the Rung-0 floor in the linear regime.
- [ ] 5.3 Unit test (tail distortion): at a saturating `m_baseline`, the measured β-space `delta` is compressed relative to the intended M-space magnitude (monotone distortion), confirming the nonlinearity is exercised.
- [ ] 5.4 Unit test (first-order inverse design): near the center the Jacobian-linearized round-trip recovers the requested latent target within tolerance.
- [ ] 5.5 Ensure tests are fast (`not slow`) and pass the repo gate: `ruff`, `mypy`, `pytest -m "not slow"`.

## 6. Driver and findings writeup

- [ ] 6.1 Add `scripts/methylation_recovery_probe.py` running the operating-point and step-scale sweeps and rendering the distortion curves (and optionally a β-space trajectory figure at a representative operating point).
- [ ] 6.2 Run the driver and write `openspec/changes/rung1-methylation-nonlinearity/findings.md` with: the operating-point sweep table/curve (`delta`/`angle` and cross-talk vs `m_baseline`, with the `none` floor), the step-scale sweep, and the absolute-distortion summary.
- [ ] 6.3 State the gate decision for Rung 2: whether `rev.logit` alone reproduces the specificity-study cross-talk (magnitude→orientation), and at what operating point it onsets, plus the exact reproduction parameters (seed set, dimensions, noise/signal scale, `m_baseline` grid, `n_components`).
