## Why

The semi-synthetic generator injects group trajectory differences along **random feature directions** that are unrelated to the features defining the stages. Instrumentation showed the resulting effects are largely invisible to MOTCO: the group signal lands off the disease-relevant subspace, and `magnitude`/`orientation` become geometrically indistinguishable. Probing InterSIM's algorithm revealed why and how to fix it: each cluster mean is `μ_i = base + δ·v_i`, where `v_i ∈ {0,1}ᵖ` is that cluster's **differential-feature indicator**. The trajectory geometry is therefore *entirely* determined by the `v_i` and `δ` — so the trajectory modes should be defined as exact operations on those indicators, not as random-direction shifts. Doing this also lets us remove the R subprocess from the runtime, which is the dominant cost and the reason generation can't be tested in CI.

## What Changes

- **BREAKING**: redefine `trajectory_mode` semantics as **feature-set surgery** on per-stage differential indicators. Group A inherits InterSIM-style random `v_i` (the baseline trajectory is intentionally *not* forced continuous — non-straight trajectories are the regime MOTCO targets); group B is a transform of A's indicators:
  - `magnitude`: same `v_i`, scaled effect `δ_B = λ·δ` → scales every step → **size/`delta`**
  - `orientation`: a **global feature permutation** `v_iᴮ = π(v_iᴬ)` (an exact isometry) → rotates the path → **`angle`**, shape/size preserved
  - `shape`: altered **step-overlaps** between consecutive stages → bends the path → **`shape`**
  - `none`: identical indicators → null
- **BREAKING**: replace the InterSIM R-subprocess generation path with a **numpy-native generator** that replicates InterSIM's generative math (`μ = base + δ·v`, per-omic covariance sampling, `rev.logit` on methylation, cross-omic coupling via the CpG→gene→protein maps). R is used **once** to export InterSIM's reference data into a cached `.npz` shipped in the repo; no R at runtime.
- Export **ground-truth per-stage/per-group differential indicators** from the generator so the showcase and study can validate that an injected mode moves its matching statistic (dominant specificity).
- Re-run the trajectory power study against the new generator/modes and **reset acceptance targets**.

## Capabilities

### New Capabilities

- `numpy-omics-generator`: numpy-native multi-omic generator that reproduces InterSIM's `μ = base + δ·v` model from cached reference data (means, covariances, cross-omic maps), with seeded reproducibility, validated realism fidelity against InterSIM, and exported differential-indicator truth.

### Modified Capabilities

- `semisynthetic-trajectory-generator`: trajectory modes redefined as feature-surgery on per-stage differential indicators; consumes the numpy generator instead of an `InterSIMResult`; emits per-stage/group indicator truth.
- `intersim-simulation-bridge`: gains a one-time reference-data export path; the runtime generation pipeline no longer depends on R.
- `simulate-command`: CLI parameter surface updated for the numpy generator and the new mode semantics.
- `trajectory-power-study`: re-run against the new generator and modes; acceptance targets reset to the new specificity/power behavior.

## Impact

- New: `src/motco/simulations/` numpy generator module + cached reference-data `.npz` (+ the R export script that produced it).
- Rewritten: trajectory-mode injection in `semisynthetic.py`; reworked params (replacing the InterSIM-subprocess + random-direction surface).
- Adapted (non-behavioral wiring): `evaluation.py`, `grid.py`, `study/*`, `showcase.py`, `cli.py` (`simulate`), and their tests.
- Removed from runtime: per-replicate R subprocess (kept only as a one-time reference-data export tool); InterSIM-dependent generation tests become R-free.
- Power study results to date are superseded; a re-run and new acceptance numbers are part of this change.
- Risks to retire during implementation: methylation additivity must live in M-value (logit) space; dominant specificity proven via the existing instrumentation before committing; numpy realism validated against InterSIM output distributions.
