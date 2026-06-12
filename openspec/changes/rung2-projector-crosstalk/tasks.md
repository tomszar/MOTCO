## 1. Test-bed module scaffold

- [x] 1.1 Create `src/motco/simulations/projector_recovery.py` with a frozen params dataclass reusing the Rung-0/Rung-1 fields (seed, n_features, per-cell sample size, noise/signal scale, manipulation `none`/`magnitude`/`orientation`, scale `c`, angle `θ`, latent `n_components`) plus a `projector` selector (`pca`/`standardize`/`plsda`/`snf`) and an optional anisotropy knob for per-feature noise heteroscedasticity.
- [x] 1.2 Implement generation by reusing the Rung-0 step construction (`a_feat`, `step_B` per manipulation) for the M-space means `μ_{g,s}` and drawing `x = μ + N(0, Σ)` from a seeded `np.random.default_rng`; reuse the Rung-0/Rung-1 helpers rather than copying them. No `rev.logit` (the feature space is the M-value frame).
- [x] 1.3 Validate params (≥2 samples per cell, `n_components` in range for the chosen projector, positive noise/signal scales, known projector selector) and raise a clear module-specific error on bad input.

## 2. Projector abstraction and measurement

- [x] 2.1 Implement the four projectors behind a uniform `(X, y) → Y` contract: `pca` (mean-centered `PCA`), `standardize` (per-feature z-score matching `evaluation.py:_concat_integration`, then `PCA`), `plsda` (`stats/pls.fit_plsda_transform` with the group label), `snf` (`stats/snf`: `get_affinity_matrix` → `SNF` → `get_spectral`). Reuse the package functions; do not reimplement.
- [x] 2.2 After projection, build the 2-group × 2-stage design via `get_model_matrix`/`build_ls_means` and the two-group contrast (identical to Rung 0/1) and call `estimate_difference` on the projected `Y`; extract `delta` and `angle` (shape is `nan` at 2 stages and dropped).
- [x] 2.3 Confirm the `pca` arm reproduces the Rung-0 floor exactly (mean-centered only, no standardization on that arm).

## 3. Comparison, sweeps, and probes

- [x] 3.1 Add a per-projector driver that runs all manipulations over the Rung-0 seed set and returns mean ± spread of `delta`/`angle` per (projector, manipulation), with `none` as the per-projector null floor and the PCA arm as the cross-projector reference.
- [x] 3.2 Add the effect-size × latent-dimensionality sweep (component count per projector; `signal_scale`/`c`/`θ`) and compute distortion metrics: absolute (measured vs intended `signal_scale·(c−1)` / `θ`) and cross-talk (magnitude→`angle`, orientation→`delta`) read against the per-projector `none` floor.
- [x] 3.3 Add the supervised-leakage probe: measure the `none` manipulation under `plsda` against the PCA `none` floor, including a small-sample / larger-component stress configuration; report any inflation.
- [x] 3.4 Report SNF results as recovery-vs-null and label absolute magnitudes embedding-relative (do not assert a unit-matched distortion ratio).

## 4. Tests

- [x] 4.1 Unit test: deterministic output for a fixed seed and projector; exactly two stages; reused Rung-0 geometry construction.
- [x] 4.2 Unit test (PCA ≈ Rung-0 floor): under `pca` with a small step, `magnitude` moves `delta` and `orientation` moves `angle` with cross-talk near the `none` baseline.
- [x] 4.3 Unit test (projector contract): each of the four projectors returns a finite `(n_samples × n_components)` latent matrix and a finite `delta`/`angle` on a small synthetic case.
- [x] 4.4 Unit test (supervised-leakage probe): the probe returns the `none` `delta`/`angle` under `plsda` and the PCA floor so leakage (if any) is quantifiable; assert the probe runs and reports both references.
- [x] 4.5 Ensure tests are fast (`not slow`) and pass the repo gate: `ruff`, `mypy`, `pytest -m "not slow"`.

## 5. Driver and findings writeup

- [x] 5.1 Add `scripts/projector_recovery_probe.py` running the per-projector comparison, the effect-size × dimensionality sweep, and the leakage probe, and rendering the distortion/cross-talk curves (and optionally a latent-space trajectory figure per projector).
- [x] 5.2 Run the driver and write `openspec/changes/rung2-projector-crosstalk/findings.md` with: the per-projector distortion/cross-talk table (with the per-projector `none` floor and the PCA reference), the effect-size/dimensionality sweep, the supervised-leakage result, and the SNF embedding-relative caveat.
- [x] 5.3 State the gate decision for Rung 3: which projector(s) are clean, whether the production `concat`/`snf` path leaks magnitude→orientation, at what effect size / component count it onsets, the exact reproduction parameters (seed set, dimensions, noise/signal scale, projector settings, component grid), and the next single factor (heterogeneous multi-omic concatenation or cross-omic coupling).
