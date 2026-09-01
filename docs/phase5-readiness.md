# Phase 5 readiness worklist

Work to complete before launching the paper-grade study. Derived from the
[Phase 4 medium PLS pilot](reports/phase4-medium-pls-pilot-2026-08-27.md) (run 2026-08-27, gate decision
**HOLD**). Every number below comes from `results/phase4-2026-08-27/report/`.

**Status: item 1 resolved 2026-09-01; items 2–5 open.** Phase 4 is complete; these are its follow-ups.
The blocking item is closed — `angle` proceeds as specified — and the 0.80 orientation power floor moves to
item 4. See [the pivotality report](reports/angle-null-pivotality-2026-09-01.md). The
[geometry audit](reports/geometry-audit-2026-09-01.md) (2026-09-01) has since narrowed items 2–4 and added
preconditions — see its remediation plan (P1–P6).

## What is already settled — do not re-litigate

These need no further pilot work and are ready for Phase 5 as-is.

- **Type I calibration.** All 18 predeclared control tests passed. Largest control rate anywhere was 0.09
  against a 0.0936 bound. Translation holds nominal level on all three statistics at every effect size.
- **Magnitude specificity and power.** Delta 1.00 throughout; angle 0.01 and shape 0.00 at the top effect,
  both *below* alpha even though magnitude's population angle is already 39.9°.
- **Shape detectability.** 0.94–1.00, monotone. The corrected Procrustes estimator finds the constructed
  bend.
- **Infrastructure.** 1,900 units, 0 failures, resumable sharding, signature-guarded resume, matched seeds,
  and the gate machinery all work. Reports regenerate byte-identically.

## 1. Diagnose the orientation power shortfall — **resolved 2026-09-01**

> **Resolved.** The non-pivotality hypothesis is **confirmed**, it fully accounts for the rejection
> inversion, and the remedy it was expected to imply is ruled out by measurement. See
> [The `angle` RRPP null is strongly non-pivotal](reports/angle-null-pivotality-2026-09-01.md)
> (run 2026-09-01, `results/angle-pivotality-2026-09-01/`, 500 replicates, 0 failures).
>
> **Decision: `angle` proceeds as specified.** Not a revised statistic, not a studentized test.

What was measured, on records that reproduce the sign-fix operating point replicate for replicate (400 of
500 identical on seeds, statistics, and p-values to
`results/orientation-signfix-2026-08-28/merged.jsonl`):

- **The association is real and large.** Each replicate's own 95th-percentile `angle` null regresses on its
  own observed angle with slope **0.811** in the orientation cell and 0.87–0.96 elsewhere; every Fisher-z
  interval excludes zero. It is specific to `angle`: under signal the same slope collapses to **0.030** for
  `delta` in the magnitude cell and **0.058** for `shape` in the orientation cell.
- **It explains the inversion.** Of 100 orientation replicates at effect 1.00, the 32 that fail to reject
  carry a larger mean observed angle (60.8° vs 46.5°) but a critical value 6.6× larger (103.6° vs 15.8°).
  Per-replicate critical values span 5.0°–176.6°. For `delta` and `shape` the critical value is flat across
  the same split.
- **No remedy recovers the power.** Within-replicate studentization is a proven no-op (it rescales both
  sides of the comparison). Cross-replicate standardization against the null controls moves orientation
  `angle` from 0.68 to **0.70** — inside Monte Carlo noise, and a diagnostic rather than a deployable test
  in any case. A single fixed threshold calibrated on the null cell drops it to **0.01**.
- **The tracking is load-bearing.** The null cell's observed angle distribution has median 5.1° and a 95th
  percentile of 153.2°. Only a replicate-specific critical value adapts to that; it is worth ~67 points of
  power relative to any fixed threshold, not a tax on power.

**What this does not settle.** The 0.80 floor. This measured one design point (n = 300, four stages, PLS at
3 latent dimensions) and cannot say whether the gap closes with more samples. That moves to item 4.

**Left open, and newly visible.** Nothing the harness persists identifies which replicates are resolvable:
within the orientation cell, selected dimensionality is 3 in 93 of 100 and CV mean AUROC is 1.0000 in every
replicate, while log(null q95) correlates +0.139 with dimensionality and −0.179 with AUROC. A direct measure
of latent trajectory-geometry stability does not exist yet.

## 2. Investigate orientation → shape at the PLS checkpoint

Orientation's `shape` rejection rate is 0.97–1.00, and localization puts it as the **only** response that
first becomes material at the PLS latent checkpoint (effects 0.75 and 1.00) rather than in the population
geometry. Every other off-diagonal is construction-present — a property of the mixed constructions Phase 2
documented, not the estimator.

**What to determine:** whether this is a property of a rank-3 stage-supervised latent space or of the shape
statistic measured within it. `simulations/specificity.py` already has the geometry probes
(`evaluate_shape_null`, `characterize_two_stage`) for a shape-free two-stage isolation.

**Decision it unblocks:** whether Phase 5 reports orientation's shape response as a known projection artifact
or as a finding about the constructions.

**Narrowed by item 1** ([report](reports/angle-null-pivotality-2026-09-01.md)): the shape response is not a
null-tracking artifact. In the orientation cell `shape` is nearly pivotal — its own critical value regresses
on its own observed statistic with slope 0.058, and is flat across the rejection split (0.0233 rejecting vs
0.0266 non-rejecting) while the observed statistic separates 6.9×. Whatever drives the 0.99 rejection rate
is in the latent geometry, not in the permutation null.

**Narrowed further by the [geometry audit](reports/geometry-audit-2026-09-01.md) (finding F3):** reflection
is ruled out as the mechanism. Allowing reflections in the latent space changes the shape distance by 0.0%
in 100 of 100 regenerated pilot replicates (the optimal alignment is already a proper rotation), and the
distance still clears the replicate's own shape-null q95 in 99 of 100. What remains is the rank-limited-projection
account: an orientation contrast that lies outside the retained rank-3 subspace re-enters the projection as
configuration deformation. The audit also found the shape statistic is reflection-*invariant* at every
pre-integration checkpoint (configuration rank < ambient dimension makes the proper-rotation constraint
vacuous), so the localization table's rows mix two contracts — resolve the policy (plan item P3) before
Phase 5 reports cross-checkpoint shape claims.

## 3. Reconsider latent dimensionality for group contrasts

Component selection saturated at **3 in all 19 cells** (range 2–3, CV AUROC 1.00). That is
`n_stages − 1`, which is the most a stage-supervised PLS-DA can carry with four stages.

The cost is measurable: the PLS reconstruction retains only ~6% of the observed orientation contrast
(cosine 0.08, norm ratio 0.06), and its captured-component top-20 precision against generator truth is 0.15
while the **observed** component's is **1.00**.

**The question:** a latent space sized to separate stage centroids is not sized to preserve the *group*
orientation contrast. If orientation is a primary estimand, the sizing criterion may need to change. This is
a design question about the architecture, not a bug — see the latent-space note in `CLAUDE.md`.

**Decision it unblocks:** the integration configuration Phase 5 commits to.

**Constrained by item 1** ([report](reports/angle-null-pivotality-2026-09-01.md)): this cannot be settled
with the diagnostics the harness records today. Within the orientation cell selected dimensionality is
effectively constant (3 in 93 of 100) and CV mean AUROC is saturated at 1.0000 in every replicate, and
neither tracks the width of the `angle` null (log q95 correlates +0.139 and −0.179 respectively). Re-sizing
the latent space on evidence needs a direct measure of latent trajectory-geometry stability, which does not
exist yet.

**Resolved by the [geometry audit](reports/geometry-audit-2026-09-01.md) (finding F1):** that measure is the
relative eigengap (λ₁−λ₂)/Σλ of the centered latent stage-mean configuration. Regenerating pivotality
replicates from persisted seeds, the pooled-configuration eigengap predicts the recorded `angle` null q95
(Spearman −0.75 across all 100 orientation replicates, −0.81 within the 16 extremes; narrow-null
replicates average gap 0.097 vs 0.046 for wide-null ones; replicates that reject average gap 0.053 vs 0.035
for those that fail). Plan item P1 persists the spectrum per replicate. Caution carried from the audit: any
group-aware sizing or supervision would void the fixed-latent-space RRPP conditioning — see the plan's
sequencing notes.

## 4. Choose the Phase 5 design point

Only after items 2–3; item 1 is resolved. The pilot used n = 300 with four stages (75 samples per
group-stage cell) over ~660 standardized features.

**What to establish:** how orientation's operating characteristics scale with samples per group-stage cell
and with feature count, so the Phase 5 sample size is chosen on evidence rather than inherited. Item 1 has
since shown the test is **not** mis-calibrated under signal, so scaling it is a meaningful measurement
rather than a more precise reading of a broken instrument.

**Also decide:** whether the 0.80 orientation power floor is achievable at a defensible sample size, or
whether the target should be revised. Per the Phase 4 exit gate, a failed target means revising the method
or the scientific claim — not merely the Monte Carlo sample size.

**This item now owns the 0.80 floor**, handed over by item 1
([report](reports/angle-null-pivotality-2026-09-01.md)), which established that the statistic and its test
are sound and that the shortfall is a design-point property. It also hands over a concrete lever: orientation
power is governed by how tightly latent trajectory geometry is determined by the data, so the design-point
study should measure how the **dispersion** of `null_summary["angle"]["q95"]` contracts with samples per
group-stage cell, not only how the rejection rate moves. In the pilot's orientation cell that dispersion
spans 5.0°–176.6°, and it — not the observed angle — decides the outcome.

**Preconditioned by the [geometry audit](reports/geometry-audit-2026-09-01.md) (findings F2, F5):** the
orientation effect axis is censored — the relocation clamp saturates the surgery at e ≈ 0.69, and 80 of 100
replicate pairs at e = 0.75 and e = 1.00 are byte-identical datasets. Fix the axis (plan item P2) before
running any design-point power study, or its top cells re-measure the same data. The audit also hands this
item a second lever besides sample size: baseline continuity (plan item P4) — n-scaling shrinks the noise
term, but the eigengap's lower tail (near-isotropic baseline draws) is what caps the curve, and that tail is
a property of the independent-indicator baseline, not of n.

## 5. Carry into the Phase 5 report contract

Small items, already evidenced, that should be settled before the study rather than discovered during it.

- **Driver reports must use the observed component**, not `pls_captured` (precision 1.00 vs 0.15).
- **Do not claim cross-replicate driver stability.** Top-20 Jaccard is 0.02–0.05 and sign agreement ~0.68,
  but that is expected: each replicate index draws a fresh indicator set, so the true driver set genuinely
  differs. Matched seeds pair cells *within* a replicate index, not across them. A stability claim needs a
  design that holds the driver set fixed.
- **`n_jobs` is part of the cell parameter signature.** RRPP seeds one RNG stream per worker, so the worker
  count changes the realized permutation draws. Phase 5 must run at the config's value; parallelize across
  shards. The sbatch script forwards `--n-jobs` only when `STUDY_N_JOBS` is set.
- **The zero-effect anchor is one measurement.** All four modes' `0.00` points resolve to a single shared
  cell and must not be counted as four independent nulls.

## Reproducing the Phase 4 evidence

```bash
# 16 resumable shards; do NOT pass --n-jobs.
for i in $(seq 0 15); do
  uv run python scripts/run_study_shard.py \
    --config examples/trajectory_power_study/phase4_pilot_100x199.json \
    --out-dir results/phase4-2026-08-27 \
    --shard-index "$i" --n-shards 16 --error-policy record &
done
wait
uv run python scripts/motco_study.py merge  --out-dir results/phase4-2026-08-27
uv run python scripts/motco_study.py report \
  --config examples/trajectory_power_study/phase4_pilot_100x199.json \
  --out-dir results/phase4-2026-08-27
```

Roughly 23 core-hours; about 90 minutes on 16 cores. The shard and merged JSONL are gitignored as
regenerable; the `report/` outputs and `PROVENANCE.txt` are committed.
