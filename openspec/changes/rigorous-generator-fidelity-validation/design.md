## Context

The numpy generator (`src/motco/simulations/generator.py`) reproduces
InterSIM's `μ = base + δ·v` model from cached reference data. Existing fidelity
evidence:

- `tests/test_generator.py::test_realism_matches_intersim_fixture` — `δ=0`
  per-feature means/variances vs a committed InterSIM fixture. `δ=0` zeroes the
  cross-omic `rho` blend and injects no effect, so it validates only the bare
  `MVN(mean, Σ)` + `rev.logit` sampling.
- A one-shot `δ=2` structural comparison (this session) over marginals, η²,
  differential-feature counts, and covariance showed strong agreement — but it
  was a single InterSIM draw at one parameter point, with no notion of
  InterSIM's own run-to-run variability.

InterSIM's RNG (Mersenne-Twister + `MASS::mvrnorm`) cannot be matched bit-for-bit
by numpy, so fidelity is necessarily a *distributional* claim. The rigorous
version compares the numpy statistic against InterSIM's *sampling distribution*.

## Goals / Non-Goals

**Goals:**
- Paper-supportable evidence that numpy reproduces InterSIM across a `delta` ×
  `p.DMP` grid, on marginals, cluster separation, differential-feature rates,
  covariance, and cross-omic coupling.
- A criterion that accounts for InterSIM's run-to-run variability (numpy inside
  InterSIM's interval), not a single-draw point comparison.
- R-free CI via committed fixtures; reproducible fixtures + a paper supplement.

**Non-Goals:**
- Not bit-exact reproduction (impossible across RNGs; not the claim).
- Not changing the generator's behavior — this is validation only.
- Not validating the trajectory modes' geometry (that is the separate
  `characterize-geometry-specificity` change).

## Decisions

- **Replicate-distribution criterion.** For each cell `(delta, p.DMP)`, run
  InterSIM `N` times to build a per-statistic distribution; run numpy `M` times.
  The numpy statistic passes if its mean (or each replicate) falls within
  InterSIM's central interval (e.g. `[q2.5, q97.5]`, or `mean ± k·sd`). Document
  the chosen interval and `N`/`M`. This is robust to RNG differences in a way the
  one-shot comparison is not.
- **Statistic battery (RNG-robust, structural).** Per omic: marginal mean/sd and
  a few quantiles; cluster separation η² (between/(between+within) variance,
  averaged over features); differential-feature count (cluster-mean range over a
  threshold) for methylation and, via derivation, expression/protein — this is
  the DMP→DEG→DEP coupling check; covariance structure (relative Frobenius of the
  sample covariance, which both should deviate from the reference identically due
  to the cluster mixture). Cross-omic coupling is captured by the η²/differential
  agreement appearing consistently across the three omics.
- **Fixtures are InterSIM summaries, not raw matrices.** Committing per-cell
  statistic distributions (a small `.npz`/CSV) keeps the repo light and CI R-free.
  The committed R script regenerates them; provenance (InterSIM version, date,
  seeds, grid) travels with the fixtures, mirroring the reference-cache pattern.
- **Grid is modest but non-degenerate.** A few `delta` values (incl. 0 as the
  existing degenerate anchor and ≥2 non-zero) × a couple `p.DMP` values. Large
  enough to be convincing, small enough to regenerate in minutes of R.
- **Reuse, don't re-derive.** Build the numpy side from the existing
  `generate_omics` + `bernoulli_indicators` + `derive_coupled_indicators` so the
  validated object is exactly the generator used by the study, not a parallel
  reimplementation.

## Risks / Trade-offs

- **Interval width is a judgement call.** Too tight → flaky CI from Monte-Carlo
  noise; too wide → weak guard. Mitigate by sizing `N`/`M` and documenting the
  interval; mark the full-grid test `slow` and keep a fast subset.
- **Covariance comparison is coarse.** Relative-Frobenius-to-reference is a
  summary; if a per-cell discrepancy appears, drill into per-omic correlation
  spectra. Acceptable for a supplement-level claim.
- **Fixture staleness.** Pinned to an InterSIM version via provenance;
  regeneration is a documented one-time R step, exactly like the reference cache.
