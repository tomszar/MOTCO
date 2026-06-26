## Context

The numpy generator constructs methylation trajectories entirely in M-value (logit) space:

```
methyl_M  ~ MVN(base_M + delta * indicator, Sigma_M)   # trajectory injected here
methylation = rev_logit(methyl_M)                        # stored as B values
```

`SemiSyntheticTrajectoryDataset.methylation` holds the B values. All three integration helpers in `evaluation.py` currently read those B values directly — the logit inversion that undoes `rev_logit` is missing. The rung ladder (Rung 1, `methylation_recovery`) proved that running any estimator on B values manufactures magnitude→orientation cross-talk because `sigmoid(c·a) ≠ c·sigmoid(a)`.

The fix is a single scalar transform applied at the top of each integration helper, before any standardisation or model fitting. Everything downstream (standardisation, SNF, PLS-DA CV) then operates in the space where the trajectory was constructed.

A clipped version of the logit already exists in `methylation_recovery.py` as `beta_to_mvalue`. The fix promotes it to `generator.py` (the natural home alongside `rev_logit`) and imports it from there everywhere else.

## Goals / Non-Goals

**Goals:**

- Apply `logit(clip=1e-6)` to `dataset.methylation` values before any integration processing.
- Add `logit` as a public function in `generator.py`, directly below `rev_logit`.
- Replace the local definition in `methylation_recovery.py` with an import alias (no public API break).
- Add `examples/trajectory_power_study/study.json` for paper-grade PLS runs.

**Non-Goals:**

- Changing `SemiSyntheticTrajectoryDataset` to store M values (conversion is at consumption, not storage — avoids touching the generator surface and its tests).
- Changing `smoke.json` (concat smoke is fine after the fix; changing it would slow the smoke).
- Modifying SNF affinity or PLS-DA CV internals.
- Fixing the independent SNF leak identified in Rung 2 (separate issue, separate change).

## Decisions

**D1 — Apply logit inside each integration helper, not in a shared pre-processing step.**

Rationale: Expression and proteomics are never logit-transformed; only methylation needs it. Doing the conversion inside each helper keeps the `integrate_semisynthetic_dataset` dispatcher clean and makes the per-layer handling self-contained. The alternative (a pre-processing hook before dispatch) would add indirection for a single-layer concern.

**D2 — Clip at `1e-6` (matching the existing `beta_to_mvalue` convention).**

Rationale: The generator draws from an MVN in M-value space and passes through `rev_logit`; values at exactly 0 or 1 are measure-zero events. The 1e-6 clip is a guard against numerical edge cases (e.g., very large positive M-values mapping to B ≈ 1) and matches what Rung 1 validated.

**D3 — Canonical `logit` lives in `generator.py`, not `evaluation.py`.**

Rationale: `generator.py` already owns `rev_logit`; `logit` is its mathematical inverse and belongs in the same module. `methylation_recovery.py` imports from `generator.py`, not the other way. This makes the pair discoverable together.

**D4 — `study.json` uses PLS integration with paper-grade sizing.**

Rationale: The rung ladder concluded PLS is the production latent-space method (additionally robust via double-CV). The smoke.json stays at `concat` for speed. A separate `study.json` with `n_replicates=500`, `permutations=999`, `n_samples=300` gives adequate Monte Carlo precision for the acceptance targets.

## Risks / Trade-offs

[Integration outputs change] → Any test that snapshots exact integration matrix values will fail. Mitigation: update those snapshots; no logic changes are needed, only reference values.

[Logit of B values near 0/1] → Without clipping, logit diverges. Mitigation: clip at 1e-6 (Decision D2); the generator's `rev_logit` never produces exact 0/1 from finite MVN draws, so this is purely defensive.

[SNF affinity on M values vs B values] → M values are unbounded reals; SNF's Gaussian kernel uses squared Euclidean distance, which is scale-sensitive. M values will tend to have larger raw magnitudes than B values (which sit in [0,1]). The affinity kernel's bandwidth (`eps`) was tuned on B values. Mitigation: acceptable for now because SNF is not the production method (PLS is); the SNF result in the study is informational. Document in the integration metadata.