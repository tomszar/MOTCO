## Context

The prior specificity work showed MOTCO's trajectory test cross-talks: a magnitude manipulation also moves orientation and shape. The diagnosed root cause is a frame mismatch — manipulations are clean similarity transforms in the generative frame (M-value space) but not in the measurement frame (β → integrated latent), driven by the methylation `rev.logit` nonlinearity and the data-dependent projector. This change builds the bottom rung of a planned ladder: a *purely linear* problem where neither nonlinearity exists, to prove the estimators recover magnitude and orientation cleanly. It is the decision gate — if recovery leaks here, the estimator/projector itself is implicated, not biology.

The estimators are reused as-is from `stats/trajectory.py`. At two stages, `_estimate_orientation` reduces to the (normalized) single step vector and `_estimate_size` to the step length, so `estimate_difference` yields `angle` = angle between the two groups' step directions and `delta` = |‖step_A‖ − ‖step_B‖| — identical in spirit to the 2-state `pair_difference`.

## Goals / Non-Goals

**Goals:**
- A deterministic, seed-reproducible Gaussian test bed that injects a *known* 2-stage feature-space geometry for two groups under MVN noise.
- Prove the clean floor: pure-magnitude → `delta`-only, pure-orientation → `angle`-only, cross-talk near null, under inline PCA.
- Exact inverse design: map a target latent step (`c·a` or `R·a`) back to the minimum-norm feature change `Δx` and expose its support/sparsity.
- A reproducible findings writeup (measured table + inverse-design recipes).

**Non-Goals:**
- No PCA integration method in `evaluation.py` (deferred to Rung 1).
- No InterSIM generator, cross-omic cascade, or `rev.logit` (Gaussian-only is Rung 1; +methylation is Rung 2).
- No PLS projector, no shape statistic, no per-feature standardization, no purity metric.
- Not a power study — this is an analytic existence proof, not a rejection-rate sweep.

## Decisions

**1. Geometry injection: explicit per-(group, stage) mean configuration + MVN noise.**
Samples are drawn as `x = μ_{g,s} + N(0, Σ)`. The trajectory step for group `g` is `step_g = μ_{g,1} − μ_{g,0}`, set directly. Group A gets a baseline step `a_feat` (a chosen feature-space direction × scale). Group B is a deterministic transform of `a_feat`:
- *magnitude:* `step_B = c · a_feat` (same direction, scaled length).
- *orientation:* `step_B = ‖a_feat‖ · (cosθ · â + sinθ · û)` where `û ⟂ â` — a length-preserving rotation by `θ` in feature space.
- *none* (null control): `step_B = a_feat`.
This makes the *intended* feature-space geometry exact before any projection. Alternative considered: drawing groups from full covariance differences (no explicit means) — rejected because the intended geometry would not be known in closed form, defeating the existence-proof purpose.

**2. No per-feature standardization at Rung 0.** PCA is fit on raw (mean-centered) features, so the map is exactly `L = Vₖᵀ` and its pseudo-inverse `L⁺ = Vₖ`, making the inverse-design formulas `Δx = Vₖ·(c−1)a` and `Δx = Vₖ·(R−I)a` exact. Standardization (which `concat` uses, and which the findings showed interacts with which features carry signal) is deliberately a Rung-1+ concern. Alternative: standardize to match the production `concat` path — rejected for Rung 0 because it muddies the exact inverse-design algebra that is the point of this rung.

**3. "Clean floor" is conditional on adequate SNR and retained components — and we say so.** PCA is orthonormal on retained components but contracts variance outside them. Recovery is clean iff the signal directions (`a_feat`, `û`) lie in the top-`k` PC subspace, which holds when the trajectory-mean spread dominates the noise `Σ` along those directions. The test bed therefore exposes the signal scale, `Σ`, and `k`, and the existence proof is stated as "clean when SNR and `k` are adequate," with the dependence documented rather than hidden by constructing the geometry to trivially live in the subspace.

**4. Measurement via `estimate_difference` (the production estimator).** We build a 2-group × 2-stage design (`get_model_matrix` / `build_ls_means` / contrast) on the PCA-projected `Y` and call `estimate_difference`, exercising the same code path the real pipeline uses, then read `delta`/`angle` (shape is `nan` at 2 stages and dropped). Cross-checking against `pair_difference` is optional and cheap.

**5. Rotation construction.** Latent-space targets use a Givens rotation `R` (k×k) in the plane spanned by `a` and a chosen orthogonal direction, by a controllable angle — orthogonal and length-preserving by construction.

**6. Reporting over seeds.** Finite samples add MC noise to the measured `delta`/`angle`. The proof reports mean ± spread over several seeds and expresses "near zero" relative to the `none` null baseline and a stated tolerance, rather than asserting exact zeros.

## Risks / Trade-offs

- [Existence proof is near-tautological if geometry is forced into the top-`k` subspace] → Expose SNR and `k`, and report recovery as a function of them so the result is "clean under adequate conditions," an informative claim, not a trivial one.
- [Inverse-design `Δx` is the minimum-norm preimage, not the only one] → State explicitly that `Δx` is minimum-norm; the support/sparsity readout is interpreted as "the most economical feature change achieving the target," which is the right object for settling the intuition.
- [MC noise could blur the near-zero cross-talk] → Average over seeds and scale tolerances to `n`; the `none` control calibrates the floor.
- [Skipping standardization diverges from the production `concat` path] → Intended: Rung 0 isolates the linear-algebra floor; standardization's interaction with signal-carrying features is explicitly a Rung-1 question.

## Open Questions

- Default `k` (PCA components) and default SNR for the headline "clean floor" run — pick values that comfortably retain the signal, then show a small sweep illustrating where recovery degrades.
- Whether the findings writeup lives in the change folder (archived with the change, matching the prior `characterize-geometry-specificity` pattern) or under `src/motco/simulations/` for longer-term visibility.
