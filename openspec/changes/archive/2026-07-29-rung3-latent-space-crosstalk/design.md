# Design — Rung 3: cross-talk through the production latent spaces

## Context

Rungs 0–2 used a generator-free test bed (known geometry injected into a clean feature space) to isolate factors abstractly. Rung 3 deliberately switches back to the **full production path**: the semi-synthetic generator → integration (latent space) → `estimate_difference` → RRPP rejection rates, as instrumented by `specificity.py`. The reason is that the cross-talk observation originated *there*, and Rung 2 has now characterized every upstream factor — what remains untested end-to-end is whether the real PLS latent space reproduces or removes the cross-talk on the actual generator.

## What the study measures

For each trajectory mode (`magnitude`, `orientation`, `shape`, `none`) the existing `evaluate_mode_specificity` already reports, over replicates, the **per-statistic RRPP rejection rate** (`delta`/`angle`/`shape` at `alpha`) plus the **group-in-stage fraction** (how much of the injected group signal lies in the stage subspace). A *specific* construction rejects predominantly on its target statistic (`magnitude`→`delta`, `orientation`→`angle`, `shape`→`shape`); cross-talk shows up as off-target rejections. Rung 3 runs this unchanged instrument through three latent spaces and compares.

## Key decisions

### Reuse the instrument; only add a latent-space selector
`evaluate_mode_specificity`/`characterize_two_stage` currently hard-default to `concat`. We add `integration_method` + `integration_params` (default `concat`, so existing callers and tests are unchanged) and forward them to **both** the RRPP evaluation and the `_group_in_stage_fraction` projection (the projection must be measured in the *same* latent space, or the diagnostic is incoherent). No change to the rejection-rate contract.

### Three latent spaces, matched everything else
Run `concat` (baseline reference), `pls` (production, stage-conditioned, double-CV-sized), and `snf` (production, graph-spectral) on **identical** seeds, effect size, `p_dmp`, `n_stages`, and permutation count. The only swept factor is the latent space — keeping the single-factor discipline of the ladder.

### Modest PLS CV knobs
PLS integration runs `plsda_doubleCV` once per replicate (not per permutation — integration builds the latent matrix once, RRPP permutes its residuals). With ~10 replicates × 4 modes that is ~40 double-CV fits; the driver passes small CV knobs (`n_repeats`, `cv2_splits`, `cv1_splits`, `max_components`) so the whole study runs in minutes locally. Effective CV params are recorded.

## Hypotheses (from Rung 2)

- **PLS:** clean `none` null (no spurious `angle`/`delta` rejection) but **attenuated orientation power** — `orientation`→`angle` rejection lower than under `concat`, because the stage-conditioned latent space compresses the group-specific direction. `magnitude`→`delta` should remain strong.
- **SNF:** magnitude↔angle cross-talk (Rung 2's ~70° leak) should surface as `magnitude` producing off-target `angle` rejections and/or unstable rates.
- **concat:** the existing baseline behavior (the cross-talk originally observed) reproduced — the reference against which PLS/SNF are read.

The point of the rung is to confirm or refute these on the *real generator with RRPP*, where multi-omic coupling and the methylation nonlinearity are present (unlike the clean test bed).

## Risks / alternatives

- **Cost** — mitigated by configurable, modest CV knobs; documented.
- **Confounds** — the production path includes generator coupling + `rev.logit`, which the test bed excluded. That is intentional here (we want the end-to-end answer), but it means a difference vs the test bed cannot be attributed to the projector alone; the findings must read PLS/SNF *relative to the concat baseline on the same generator*, not against the Rung-2 test-bed numbers.
- **Alternative** — extend the grid/power study instead of the specificity instrument. Rejected for now: the specificity rejection-rate instrument is the lighter, purpose-built diagnostic and already encodes the target-statistic logic; a full power grid is a later step if the specificity comparison warrants it.
