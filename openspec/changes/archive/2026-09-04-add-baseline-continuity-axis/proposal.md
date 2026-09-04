# add-baseline-continuity-axis

## Why

The geometry audit's synthesis finding (F5, [geometry-audit-2026-09-01](../../../docs/reports/geometry-audit-2026-09-01.md), plan item P4) identified the independent-indicator baseline as the deepest layer of the orientation power shortfall: group A draws each stage's differential program independently, so the stage means form a near-regular simplex with a near-zero eigengap — the trajectory has no direction to differ in, PC1 is an ill-determined estimand per draw, and the `angle` null width tracks that anisotropy (F1, Spearman −0.75/−0.81). n-scaling attacks only the noise term; the eigengap's lower tail is a property of the baseline construction, not of n. This is the change the audit judged "most likely to move the orientation operating characteristics by tens of points," and Phase-5 readiness item 4 (the 0.80 power floor and the design-point choice) is blocked on it.

## What Changes

- Add a **baseline stage-program continuity parameter** to `SemiSyntheticTrajectoryParams` (`baseline_continuity: float = 0.0`, ρ ∈ [0, 1)): group A's per-CpG methylation indicators follow a stationary first-order Markov chain across stages (x₁ ~ Bern(p_dmp); P(1→1) = p + ρ(1−p), P(0→1) = p(1−ρ)) instead of independent per-stage draws. The per-stage Bernoulli(p_dmp) marginal is preserved at every ρ, so per-stage indicator counts, δ semantics, the CpG→gene→protein coupling derivation, and the meaning of `p_dmp` are unchanged along the axis; cross-stage indicator correlation is ρ^|t−s|, giving stage-mean configurations that trend (pairwise distances grow with stage separation) and carry a well-conditioned PC1.
- **ρ = 0 reproduces the current independent baseline byte-identically** (same uniform draw block; both thresholds collapse to p_dmp), keeping the isotropic case as the declared stress-test endpoint.
- **Generalize `expected_surgery_headroom`** from the independence union probability `1 − (1−p)^n` to the Markov union probability `1 − (1−p)·(1 − p(1−ρ))^(n−1)`, so pool-limited surgery headroom (and study enumeration's fail-loud rejection) stays correct along the axis. Pools grow with ρ for orientation and shape-relocate — stage programs overlap more, shrinking the active union.
- **Truth metadata records `baseline_continuity`** alongside the existing generator parameters.
- **Study reporting resolves orientation operating characteristics by baseline continuity**: when a study sweeps `generator.baseline_continuity` (the existing generic axis machinery already permits it), the report presents orientation power and the eigengap distribution as functions of the continuity axis, making "MOTCO resolves orientation differences when the trajectory has a direction to differ in" a designed, monotone operating characteristic with the persisted `config_spectrum` eigengap (P1) as the observable that carries the claim to real data.
- **BREAKING (by policy, not accident):** the new generator field enters `parameter_signature`, so pre-change shards refuse to resume — consistent with the established signature-guard policy (`SHAPE_STATISTIC_VERSION` precedent). Historical committed results keep their recorded values.

Out of scope: running the Phase-5 design-point study itself; any change to the surgery transforms, the statistics, or the RRPP test; group-aware latent-space supervision (explicitly cautioned against by the audit's sequencing notes).

## Capabilities

### New Capabilities

(none)

### Modified Capabilities

- `semisynthetic-trajectory-generator`: the baseline-indicator requirement gains the continuity axis (stationary Markov persistence with preserved marginals, ρ = 0 identical to today's draws, continuity recorded as truth), and the surgery-headroom expectation is defined under the continuity-adjusted union probability.
- `trajectory-power-study`: headroom enumeration remains valid along the continuity axis, and reporting presents orientation operating characteristics resolved by baseline continuity with the eigengap as the linking observable.

## Impact

- `src/motco/simulations/generator.py` — `bernoulli_indicators` gains the Markov variant (or a sibling `markov_indicators`); ρ = 0 path must consume the identical RNG stream.
- `src/motco/simulations/semisynthetic.py` — new params field + validation, `_baseline_methyl`, `expected_surgery_headroom`, truth block.
- `src/motco/simulations/study/` — enumeration headroom check picks up the generalized formula automatically; report gains the continuity-resolved orientation view (analogous to `eigengap_stratified_power.csv`).
- `src/motco/simulations/grid.py` — no code change needed for the signature (params are hashed generically), but the resume break is a documented consequence.
- Tests — marginal preservation, ρ = 0 byte-identity, cross-stage correlation, eigengap monotonicity in expectation, headroom formula against Monte Carlo, truth recording, enumeration along the axis.
- Docs — `CLAUDE.md` semisynthetic bullet, module docstring ("intentionally not forced continuous" becomes the ρ = 0 endpoint of a declared axis), `docs/phase5-readiness.md` item 4 hand-off.
