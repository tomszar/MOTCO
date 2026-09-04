## Why

The tested trajectory `angle` is a **principal-axis divergence** — PC1 of each group's centered stage-mean configuration, signed by net displacement — but `stats/attribution.py` explains **per-adjacent-transition contrasts** (`d/‖d‖` per stage transition). The two coincide only for two stages; for k ≥ 3 a significant `angle` is interpreted through a decomposition of a related but different quantity (geometry audit 2026-09-01, finding F4), and the roadmap's Phase-3 promise to extend attribution with "signed principal orientations" was never implemented. Phase 6 will interpret significant orientation results through attribution, so the estimand and its decomposition must match before then.

## What Changes

- Add a **principal-orientation component** to `analyze_orientation_attribution`: per group, PC1 of the centered stage-mean configuration in feature space — computed for the observed means and the PLS-reconstructed means — signed by net displacement with the identical convention as `_estimate_orientation`, and the group contrast decomposed per feature into observed, PLS-captured, and residual parts alongside the existing per-transition contrasts.
- Gate the principal contrast with an **eigengap degeneracy qualifier**: report each group's relative eigengap (reusing `configuration_spectra` / `configuration_spectrum` from `stats/trajectory.py`) and mark the principal contrast as degenerate below a documented threshold instead of silently emitting a noise-driven axis (audit finding F1's failure mode).
- Extend the **bootstrap stability** machinery to the principal component, with the sign anchored per replicate by the same net-displacement convention so resampling cannot flip the axis arbitrarily.
- Pin the **k = 2 equivalence**: with exactly two stages the principal-orientation contrast SHALL equal the single transition contrast exactly (rank-1 configuration ⇒ PC1 is the transition direction) — an estimator-consistency contract tested against `_estimate_orientation` on the same means.
- Sharpen the **interpretation boundary**: the result states that the principal-orientation contrast decomposes the tested `angle` estimand while per-transition contrasts describe where along the trajectory the groups diverge.
- Align documentation: narrow the `angle` estimand wording to "principal-axis divergence" in the orientation-invariance spec purpose and known-angle scenario, and correct the roadmap bullet (`docs/roadmap.md:133`) from promise to delivered.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `orientation-driver-attribution`: adds a requirement for the principal-orientation contrast (estimator-consistent PC1, observed/PLS-captured/residual decomposition, degeneracy qualifier, k = 2 equivalence, bootstrap stability) and modifies the interpretation-boundary requirement to state which quantity each decomposition explains.
- `trajectory-orientation-invariance`: narrows the documented estimand — the purpose and the "known angle is recovered" scenario are stated in terms of principal-axis divergence rather than an unqualified "direction of progression" (they coincide only for straight or two-stage trajectories). No estimator behavior changes.

## Impact

- **Code:** `src/motco/stats/attribution.py` (new dataclass fields/records + tables for the principal component; `OrientationAttributionResult` gains fields — additive, existing fields unchanged), reuse of `configuration_spectrum` from `src/motco/stats/trajectory.py`; no change to `estimate_difference`, RRPP, or any tested statistic.
- **Tests:** `tests/test_attribution.py` — k = 2 equivalence, k ≥ 3 estimator consistency against `_estimate_orientation`, degeneracy gating, bootstrap sign stability, frame schema.
- **Downstream:** `src/motco/simulations/attribution_diagnostics.py` consumes the result — verify additive compatibility; `attribution_frames` / `write_attribution_outputs` gain a principal-orientation table.
- **Docs:** `docs/roadmap.md:133`, attribution module docstring, orientation estimator cross-references. No study records, signatures, or persisted schemas are touched.
