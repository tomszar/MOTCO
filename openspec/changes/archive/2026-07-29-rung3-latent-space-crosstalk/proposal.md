## Why

Rungs 0–2 isolated the trajectory cross-talk question one factor at a time on a **generator-free test bed**: Rung 0 proved the linear floor is clean under PCA; Rung 1 showed the methylation `rev.logit` is invertible and not a standing cross-talk source; Rung 2 swept the projector and found per-feature `standardize`/`concat` clean, `snf` leaky (magnitude→~70° angle), and stage-conditioned PLS-DA non-leaking but **lossy** (compresses orientation at low component counts).

But the cross-talk the whole ladder is chasing was first observed in the **dominant-specificity study** (`specificity.py`) — which runs the *full production generator + RRPP rejection rates*, not the injected-geometry test bed — and that study has only ever been run through the **`concat` baseline**. PR #23 established that `concat` is a *baseline/diagnostic*, not a production latent space: the production latent spaces are **SNF** (graph-spectral) and the newly-implemented **PLS** (stage-conditioned, double-CV-sized). So the specificity-study cross-talk has never been measured through a real production latent space.

This rung asks the direct end-to-end question: **does the per-statistic cross-talk reproduce through the production PLS latent space, or was it a `concat`-baseline artifact?** Rung 2 predicts a specific signature for PLS — a clean null (no false orientation) but *attenuated orientation power* (the latent space compresses the group-specific direction) — which this rung tests on the real generator with RRPP.

This **supersedes** Rung 2's gate decision ("Rung 3 = heterogeneous multi-omic concatenation"). That plan targeted a property of the `concat` baseline; with `concat` reclassified as a baseline and the real PLS latent space now available, the higher-value question is cross-talk through the production latent spaces, not through the baseline's concatenation geometry.

## What Changes

- **Make the specificity instrumentation latent-space-selectable.** Thread `integration_method` / `integration_params` through `evaluate_mode_specificity` and `characterize_two_stage` (default `concat` for backward compatibility), so the per-statistic RRPP rejection-rate study can run through `concat`, `snf`, or `pls`. The group-in-stage projection (`_group_in_stage_fraction`) uses the same selected latent space.
- **Add a Rung-3 driver** that runs each trajectory mode (`magnitude`/`orientation`/`shape`/`none`) through the three latent spaces on matched seeds and effect size, reporting the per-statistic (`delta`/`angle`/`shape`) rejection-rate table and the group-in-stage fraction per latent space — with modest PLS cross-validation knobs (double-CV per replicate is the cost driver).
- **Findings writeup** (matching the Rung-0/1/2 pattern): the per-latent-space rejection-rate tables; whether magnitude→orientation (and shape) cross-talk reproduces through PLS; confirmation or refutation of Rung 2's "PLS clean-null but lossy" prediction on the real generator; and the gate decision for the next rung.

## Capabilities

### New Capabilities
- `latent-space-crosstalk-study`: Run the dominant-specificity per-statistic RRPP rejection-rate study through a *selectable production latent space* (`concat` baseline, `snf`, or `pls`) on the full semi-synthetic generator, and compare per-mode cross-talk and statistic-specificity across latent spaces.

### Modified Capabilities
<!-- None: the specificity instrumentation gains an integration selector, but its rejection-rate contract is unchanged; this is additive. -->

## Impact

- **Modified code:** `src/motco/simulations/specificity.py` — `integration_method`/`integration_params` parameters on `evaluate_mode_specificity` and `characterize_two_stage`, forwarded to both the evaluation and the group-in-stage projection.
- **New code:** a Rung-3 driver under `scripts/` (e.g. `latent_space_crosstalk_probe.py`); unit tests under `tests/` for the threading (each latent space runs and is recorded).
- **New docs:** `findings.md` in the change folder.
- **Reused, unchanged:** the generator, `evaluation.py` integration methods (incl. the new `pls`), `stats/trajectory`, `stats/permutation`.
- **Dependencies:** none new.
- **Depends on:** PR #23 (`feat/pls-latent-integration`) — this branch is stacked on it; merge #23 first.
- **Cost note:** PLS integration runs double-CV once per replicate; the driver exposes CV knobs and uses modest defaults so the study runs locally in minutes, not a cluster job.
- **Out of scope:** changing the generator, estimators, or PLS integration behavior; CLI/grid wiring of PLS (separate follow-up); a full Type I / power grid (this is the diagnostic specificity comparison at modest replicate counts).
