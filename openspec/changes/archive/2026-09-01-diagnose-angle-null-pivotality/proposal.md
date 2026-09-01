> **Status: confirmed — proceed.** This change's hypothesis was briefly thought to be displaced by an
> orientation sign-anchor defect (`../fix-orientation-sign-anchor/`). That defect was real and is fixed, but
> the corrected re-run of 2026-08-28 shows it is **not** the cause: orientation `angle` power rose only from
> 0.65 to 0.68 against the 0.80 floor, the curve is still flat, and the near-180° null mass is unchanged at
> 21 of 700 null replicates. See `docs/reports/orientation-sign-anchor-2026-08-28.md`. The shortfall and the
> rejection inversion survive correction, so the non-pivotality hypothesis stands and this change proceeds
> as written.

## Why

Phase 4's medium PLS pilot passed every Type I control and both the magnitude and shape power targets, but orientation `angle` power stalled at 0.65 against the preregistered 0.80 floor and was nearly flat across effect sizes (0.59 → 0.65). The cause is unidentified and the obvious explanation is contradicted by the records: at orientation effect 1.00 the replicates that *fail* to reject carry a **larger** mean observed latent angle (66.7°) than those that reject (52.5°), an inversion that also appears in shape at effect 1.00 (83.3° vs 61.5°) but not in the translation null (30.7° rejecting vs 7.8° non-rejecting). Latent dimensionality, weak construction, cross-talk, and Monte Carlo noise are already ruled out, so this is the blocking item on the Phase 5 readiness worklist — Phase 5 cannot commit to the `angle` test until it is explained.

The leading hypothesis is that the RRPP permutation null for `angle` is **not pivotal**: it co-varies with the observed statistic within a replicate, so a replicate whose latent geometry inflates the observed angle inflates its own null at least as much. The Phase 4 pilot cannot settle this because `include_null_distributions` was off and, more fundamentally, because nothing about the permutation null survives into the persisted replicate record at all.

## What Changes

- Summarize each replicate's RRPP permutation null into a compact, JSON-safe per-statistic record — count, mean, standard deviation, and the quantiles needed to locate the observed statistic against its own null (including the alpha-level critical value) — computed from the null draws the harness already produces whenever `permutations > 0`.
- Persist that summary through the grid's replicate record so it reaches the study JSONL. Today `SimulationEvaluationParams.include_null_distributions` yields null vectors inside evaluation, but `SimulationReplicateResult` has no field for them and silently drops them; `include_null_distributions` continues to gate retention of the **full** null vectors, which stay out of persistence.
- Keep both additions strictly additive: records written before this change still load, and no parameter that affects generation, integration, permutation draws, or the cell parameter signature changes — so an existing shard remains resumable and its realized draws are unchanged.
- Add a committed orientation pivotality diagnostic profile: a modest replicate set at orientation effect 1.00, with the matched translation-null and magnitude comparators needed to interpret it, run through the existing study runner.
- Add a diagnostic analysis that reads merged study records and quantifies, across replicates within a cell, the association between each replicate's observed statistic and its own null location and spread — reported per statistic, with the rejecting/non-rejecting split that motivated the question and a standardized (studentized) recomputation of the test as the counterfactual.
- Produce a dated findings report under `docs/reports/` that states whether the `angle` null is pivotal under signal, and record the consequence for Phase 5 in `docs/roadmap.md` and `docs/phase5-readiness.md`: the `angle` test proceeds as specified, is replaced by a pivotal or studentized statistic, or carries a revised power target.

## Capabilities

### New Capabilities

- `angle-null-pivotality-diagnostic`: The protocol that decides whether MOTCO's RRPP `angle` test is pivotal under signal — a reproducible diagnostic run, a per-replicate association analysis between the observed statistic and its own permutation null, a studentized counterfactual, and a dated finding that resolves the Phase 5 blocking gate.

### Modified Capabilities

- `simulation-evaluation-harness`: Evaluation results carry a compact per-statistic summary of the permutation null whenever permutations are run, distinct from the existing opt-in retention of the full null distributions.
- `simulation-grid-orchestration`: Persisted replicate records carry that null summary, additively and without changing the cell parameter signature.

## Impact

- Extends the evaluation result contract in `src/motco/simulations/evaluation.py` and the persisted record in `src/motco/simulations/grid.py`; no change to trajectory statistics, RRPP, the generator, integration, or public defaults.
- Adds a diagnostic config under `examples/trajectory_power_study/` and an analysis entry point under `scripts/`, following the existing study-script conventions.
- Adds a findings report under `docs/reports/` and updates `docs/roadmap.md` (Phase 4 gate follow-up) and `docs/phase5-readiness.md` (item 1 resolution).
- Existing study configs, reports, and the archived Phase 4 evidence are unchanged; the committed Phase 4 `report/` outputs still regenerate byte-identically from their own records.
- No new runtime dependency; the diagnostic runs from cached reference data without R.
