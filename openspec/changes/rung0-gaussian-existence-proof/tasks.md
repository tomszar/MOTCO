## 1. Test-bed module scaffold

- [x] 1.1 Create `src/motco/simulations/linear_recovery.py` with a frozen params dataclass (seed, n_features, per-group/stage sample sizes, covariance spec, signal scale, `a_feat` direction spec, manipulation = `none`/`magnitude`/`orientation`, scale `c`, rotation angle `θ`, PCA `n_components`) and a dataset dataclass (feature matrix + group/stage labels + the known feature-space step vectors).
- [x] 1.2 Implement deterministic generation: draw `x = μ_{g,s} + N(0, Σ)` from a seeded `np.random.default_rng`, with exactly two stages per group; build group A's baseline step `a_feat` and group B's step per manipulation (`c·a_feat`; length-preserving rotation `cosθ·â + sinθ·û`; or identical for `none`).
- [x] 1.3 Validate params (≥2 samples per group/stage cell, `n_components` ≤ rank, non-negative scales, `Σ` SPD) and raise a clear module-specific error on bad input.

## 2. Linear projection and measurement

- [x] 2.1 Fit inline PCA (mean-centered, no per-feature standardization) on the pooled feature matrix; retain `n_components`; expose loadings `Vₖ` for inverse design.
- [x] 2.2 Project the feature matrix to latent coordinates and build the 2-group × 2-stage design via `get_model_matrix` / `build_ls_means` and the two-group contrast.
- [x] 2.3 Call `estimate_difference` on the projected `Y`; extract `delta` and `angle` for the group pair (shape is `nan` at 2 stages and dropped). Optionally cross-check `angle`/`delta` against `pair_difference`.

## 3. Exact inverse design

- [x] 3.1 Implement `inverse_design_magnitude(a, Vk, c)` returning `Δx = Vₖ·(c−1)·a`, and assert `L·Δx ≈ (c−1)·a` within numerical tolerance.
- [x] 3.2 Implement a Givens rotation builder `R(plane, θ)` (k×k, orthogonal) and `inverse_design_orientation(a, Vk, R)` returning `Δx = Vₖ·(R−I)·a`, asserting `L·Δx ≈ (R−I)·a`.
- [x] 3.3 Add a `Δx` readout: support (indices above a relative threshold), a sparsity/concentration summary (e.g. participation ratio or top-k mass), and overlap of the magnitude vs orientation supports.

## 4. Existence-proof driver and seeds

- [x] 4.1 Add a driver that runs `none`/`magnitude`/`orientation` over several seeds and returns mean ± spread of measured `delta`/`angle` per manipulation, with `none` as the null floor.
- [x] 4.2 Add a k=3..10 sweep showing how PCA component count affects recovery and the angle noise floor. (SNR sweep deferred to Rung 1.)

## 5. Tests

- [x] 5.1 Unit test: deterministic output for a fixed seed; exactly two stages; per-cell means approximate the spec.
- [x] 5.2 Unit test (clean floor): under adequate SNR/`k`, `magnitude` moves `delta` with `angle` near the `none` baseline (within tolerance); `orientation` moves `angle` with `delta` near baseline.
- [x] 5.3 Unit test (inverse design): projected `Δx` recovers the requested latent target `(c−1)a` / `(R−I)a` within tolerance for both magnitude and orientation.
- [x] 5.4 Ensure tests are fast (`not slow`) and pass the repo gate: `ruff`, `mypy`, `pytest -m "not slow"`.

## 6. Findings writeup

- [x] 6.1 Run the driver and write `openspec/changes/rung0-gaussian-existence-proof/findings.md` with the intended-geometry→measured table (per manipulation: measured `delta`/`angle`, with the `none` floor) and the k=3..10 dependence.
- [x] 6.2 Record the inverse-design feature recipes: magnitude vs orientation `Δx` support/sparsity summaries and their overlap, and state whether the result supports "magnitude = same features scaled / orientation = different features" or "orientation mixes features within the loading subspace."
- [x] 6.3 Note the exact reproduction parameters (seed set, dimensions, covariance, signal scale, `n_components`) and the verdict on whether the Rung-0 floor is clean (the gate decision for Rungs 1–2).
