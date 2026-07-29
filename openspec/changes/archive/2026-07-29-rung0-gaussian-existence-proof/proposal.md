## Why

MOTCO claims its trajectory-geometry test recovers three distinct facets — magnitude (`delta`), orientation (`angle`), and shape — but the prior specificity work (`archive/2026-06-04-characterize-geometry-specificity/findings.md`) showed heavy cross-talk: magnitude leaks into orientation and shape. That work also identified the likely root: manipulations are clean similarity transforms in the *generative* frame (M-value space) but not in the *measurement* frame (β → integrated latent space), because of the methylation `rev.logit` nonlinearity and the data-dependent projector. Before characterizing how realistic conditions degrade recovery, we need a **clean floor**: proof that on a *purely linear* problem the estimators recover magnitude and orientation cleanly. This is the bottom rung of a planned ladder and the decision gate for the rungs above — if recovery leaks here, the problem is the estimator/projector itself, not biology.

## What Changes

- Add a self-contained, analytic **Rung-0 Gaussian test bed**: draw features `X ~ MVN(known Σ)`, inject a *known* 2-stage trajectory geometry directly in feature space for two groups, project with PCA (sklearn, inline — **not** the evaluation harness), and measure `delta`/`angle` via the existing `stats/trajectory.py` estimators. No InterSIM generator, no cross-omic cascade, no `rev.logit`.
- **Stage = 2 only**, so Procrustes `shape` is degenerate and excluded — the scope is strictly magnitude and orientation.
- **Existence proof (clean floor):** demonstrate that under a pure linear projector a pure-magnitude manipulation registers as `delta`-only and a pure-orientation manipulation as `angle`-only, with cross-talk near zero.
- **Exact inverse design (settle the feature intuition):** given group A's latent step `a` and the PCA map `L` (loadings `Vₖ`), compute the minimum-norm feature-space change `Δx` that produces a target pure-magnitude latent step (`b = c·a`, `Δx = Vₖ·(c−1)a`) and a target pure-orientation latent step (`b = R·a`, `Δx = Vₖ·(R−I)a`); inspect the support/sparsity of `Δx` to test whether "magnitude = same features scaled" and "orientation = different features change" hold, or whether orientation is a feature-mixing rotation within the loading subspace.
- Add a committed **findings writeup** with the intended-geometry→measured table and the inverse-design feature recipes.
- **Out of scope (deferred to later proposals):** the PCA integration method in `evaluation.py`, the Gaussian-only cascade-bypass generation path, the +methylation rung, and the PCA→PLS projector swap. No purity metric.

## Capabilities

### New Capabilities
- `linear-geometry-recovery`: An analytic, generator-free Rung-0 test bed that validates clean recovery of trajectory magnitude and orientation under a linear (PCA) projector at two stages, and an exact inverse-design tool that maps a target latent step geometry back to the minimum-norm feature-space change for inspection.

### Modified Capabilities
<!-- None: this is a self-contained new test bed; it does not change requirements of the evaluation harness, generator, or trajectory estimators. -->

## Impact

- **New code:** a test-bed module under `src/motco/simulations/` (e.g. `linear_recovery.py`) plus its unit tests under `tests/`.
- **New docs:** a findings writeup (location alongside the existing geometry findings, e.g. under the change folder or `simulations/`).
- **Reused, unchanged:** `stats/trajectory.py` estimators (`estimate_difference` / `_estimate_size` / `_estimate_orientation`, or the 2-state `pair_difference`).
- **Dependencies:** `numpy`, `scikit-learn` (PCA) — both already present.
- **No changes** to `evaluation.py`, `semisynthetic.py`, `generator.py`, or any existing spec.
