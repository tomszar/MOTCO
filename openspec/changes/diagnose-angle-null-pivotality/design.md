> **Status: confirmed — proceed.** This change's hypothesis was briefly thought to be displaced by an
> orientation sign-anchor defect (`../fix-orientation-sign-anchor/`). That defect was real and is fixed, but
> the corrected re-run of 2026-08-28 shows it is **not** the cause: orientation `angle` power rose only from
> 0.65 to 0.68 against the 0.80 floor, the curve is still flat, and the near-180° null mass is unchanged at
> 21 of 700 null replicates. See `docs/reports/orientation-sign-anchor-2026-08-28.md`. The shortfall and the
> rejection inversion survive correction, so the non-pivotality hypothesis stands and this change proceeds
> as written.

## Context

See `proposal.md` — Why. Three facts about the current code shape the approach:

- `evaluation.py` already builds the full per-statistic null vectors on every RRPP run (`_extract_null_distributions`) and uses them for the plus-one empirical p-value. `include_null_distributions` only controls whether those vectors are *returned*; they are always computed.
- `grid.py`'s `SimulationReplicateResult` has no field for anything null-related, so even when the vectors are returned they are dropped before JSONL persistence. This is the whole reason Phase 4 cannot answer the question from its own records.
- `parameter_signature(cell)` is derived from the cell's generator and evaluation parameters and guards resume. Phase 4 established that `n_jobs` belongs in the signature because it changes realized permutation draws.

One framing correction that drives the design. The readiness worklist calls the suspected problem "mis-calibration". It is not: Phase 4 passed all 18 predeclared Type I controls, and RRPP p-values are exact under exchangeability regardless of whether the statistic is pivotal. Non-pivotality does not break validity — it costs **power**, because a replicate whose latent geometry inflates the observed angle inflates its own critical value with it, so the signal never clears its own bar. The diagnostic is therefore a power-mechanism investigation, and its acceptance evidence is a power comparison, not a Type I comparison.

## Goals / Non-Goals

**Goals:**

- Make every future replicate record self-describing enough to answer "did this replicate's null move with its observed statistic?" without a re-run — including Phase 5's own records.
- Settle the pivotality question for `angle` with a run small enough to sit beside the Phase 4 evidence rather than compete with it.
- Produce a decision, not a description: one of three named Phase 5 consequences.

**Non-Goals:**

- Designing or shipping a replacement statistic. If the diagnostic says a pivotal statistic is needed, that is the *next* change; this one names the need and the evidence for it.
- Persisting full null vectors. The compact summary is designed to be sufficient for this question; storing 999 floats × 3 statistics × 500 replicates × 19 cells is not.
- Re-running or re-reporting Phase 4. Its committed `report/` outputs must still regenerate byte-identically from their own records.

## Decisions

### The null summary is unconditional, not opt-in

Compute the summary on every RRPP run and persist it always, rather than gating it behind `include_null_distributions` or a new flag.

*Why:* the null draws already exist at that point; summarizing them is a sort over at most a few thousand floats, immeasurable against a PLS double-CV fit. Gating it would have exactly reproduced the Phase 4 failure mode — the question was unanswerable because a diagnostic switch was off, and the same class of question will recur in Phase 5. An unconditional summary makes Phase 5's records answer it for free.

*Alternative rejected:* a new `include_null_summary` parameter. It would enter the parameter signature (or dishonestly not), split the record population into two kinds, and buy nothing but a few bytes.

### The summary does not enter the parameter signature

*Why:* the signature exists to prevent unsafe resume — a mismatch means the records were produced under different *behavior*. Summarizing draws that already happened changes no generation, integration, or permutation behavior, and no RNG stream. Adding it to the signature would invalidate every existing shard for a purely additive record field. A test must pin this: same cell, same seed, same signature, identical p-values before and after.

### Contents: count, mean, sd, and quantiles including the alpha critical value

Per statistic: retained draw count, mean, standard deviation, and the q50/q90/q95/q99 quantiles, with non-finite draws excluded and the retained count reported so exclusions are visible. The observed statistic is already persisted in `pair_statistics`, so the pair (observed, its own q95) — the quantity the whole hypothesis is about — becomes directly computable from one record.

*Why quantiles and not just mean/sd:* the RRPP null for an angle is bounded and skewed; the critical value is what the test actually compares against, and it is not recoverable from mean and sd under a non-normal null.

### The counterfactual must be cross-replicate, because within-replicate studentization is a no-op

This is the load-bearing statistical decision. Standardizing the observed statistic by its own null's mean and sd, and comparing it to that same null's quantiles, is a strictly monotone transform applied to both sides — **the p-value is unchanged**. Within-replicate studentization cannot recover a single rejection.

The counterfactual is therefore a *cross-replicate recalibration*: form `z = (observed − null_mean) / null_sd` per replicate, then calibrate `z` against the distribution of `z` over the null-control cells' replicates, and report the resulting rejection rate beside the as-specified rate for every cell. If non-pivotality is the mechanism, `z` separates the orientation cell from the null cells far better than the raw statistic does, and this shows it directly from the recorded summaries — with no additional permutations.

*Consequence to state plainly in the report:* this counterfactual is a **diagnostic, not a deployable test**. It borrows a null-mode reference distribution that does not exist in real data. A production pivotal test would need a nested/double permutation or an analytically studentized statistic, which is out of scope here (see Non-Goals). Reporting it as a candidate production test would be a scientific error.

### Diagnostic profile: three cells at effect 1.00, Phase 4 parameters otherwise unchanged

`orientation` (the shortfall), `translation` (the null control that behaved normally — 30.7° rejecting vs 7.8° non-rejecting), and `magnitude` (the comparator whose statistic reached power 1.00 and is presumed pivotal enough). Everything else — n = 300, four stages, PLS on M-values, 199 permutations, `n_jobs = 1` — matches `phase4_pilot_100x199.json` exactly.

*Why hold the parameters fixed:* the target of the diagnostic is the Phase 4 behavior itself. Changing the permutation count or sample size would explain a different configuration's behavior. At roughly 44 core-seconds per unit (23 core-hours / 1,900 units), 3 cells × 100 replicates ≈ 3.7 core-hours — under 15 minutes on 16 cores.

*Why include `magnitude`:* an association between observed statistic and null found only in orientation is a finding; one found everywhere including the comparator is a property of RRPP under any signal and would reframe the conclusion.

### Analysis lives in a testable module with a thin script wrapper

`src/motco/simulations/pivotality.py` (association measures, rejection split, cross-replicate counterfactual, table construction) plus `scripts/angle_null_pivotality.py` as the entry point — the same split as `specificity.py` / `scripts/geometry_specificity_probe.py`. The analysis reads merged JSONL through the existing `read_replicate_results`, so it is R-free, re-runnable, and unit-testable on synthetic records without running a study.

*Alternative rejected:* extending `study/report.py`. This is a one-question diagnostic, not part of the study report contract; folding it in would put a temporary investigation into the Phase 5 reporting path.

## Risks / Trade-offs

- **The association may be present in all three cells** → then non-pivotality is a general RRPP property, not orientation-specific, and it fails to explain why *only* angle fell short. The report must then say the hypothesis is not supported and hand the readiness worklist a still-open item 1 rather than a false resolution. The comparator cell exists precisely to make this outcome visible instead of invisible.
- **100 replicates per cell gives a coarse association estimate** → the diagnostic asks for a direction and rough magnitude of association, not a precise coefficient; a null-tracking mechanism strong enough to cost 15 points of power is not a subtle correlation. Report an uncertainty measure alongside every association so a weak result is not over-read.
- **Unconditional summarization changes every future record** → mitigated by making the field additive with an empty default and testing that pre-change records still load; no existing file is rewritten.
- **Diagnostic and pilot records both live under `results/`** → keep the diagnostic in its own dated output directory so nothing can be merged into or confused with the Phase 4 record set.
- **The counterfactual is tempting to over-claim** → the spec requires reporting its null-cell rejection rate, and the report must carry the "diagnostic, not deployable" statement explicitly.

## Migration Plan

Purely additive; no data migration. New fields default to empty, `_replicate_result_from_dict` tolerates their absence, and the parameter signature is unchanged, so existing shards stay resumable and the committed Phase 4 `report/` outputs still regenerate byte-identically. Rollback is reverting the code: the new record fields are ignored by every existing reader.
