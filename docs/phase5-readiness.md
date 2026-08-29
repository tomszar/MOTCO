# Phase 5 readiness worklist

Work to complete before launching the paper-grade study. Derived from the
[Phase 4 medium PLS pilot](reports/phase4-medium-pls-pilot-2026-08-27.md) (run 2026-08-27, gate decision
**HOLD**). Every number below comes from `results/phase4-2026-08-27/report/`.

**Status: not started.** Phase 4 is complete; these are its follow-ups.

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

## 1. Diagnose the orientation power shortfall — **blocking**

> **Updated 2026-08-28.** An orientation sign-anchor defect was found and fixed
> ([report](reports/orientation-sign-anchor-2026-08-28.md)): PC1's sign was anchored on the centered
> first-stage row, which flips for bent trajectories. It was a real bug, but **it does not explain this
> item.** On byte-identical data, corrected power is **0.64/0.64/0.67/0.68** across effects 0.25 → 1.00 —
> up ~3 points, still flat, still short of 0.80 by more than two standard errors. The near-180° null mass is
> **unchanged at 21/700**, so the artifact in the study path has a driver the fix does not address. The
> figures below are the pre-fix values; substitute the corrected ones above when acting on this item.
>
> One finding narrows the search: of 700 null replicates, 46 flip to their supplement under the corrected
> anchor, and exactly 21 cross into >150° while 21 cross out — a wash. The sign is still being noise-decided
> in the latent space, consistent with `PC1 · (stage_last − stage_first)` itself landing near zero, or with
> PC1's *direction* being unstable when the configuration's top two singular values are close.

Orientation `angle` power is 0.65 against the 0.80 floor and nearly flat across effects (0.59 → 0.65).
**The cause is not identified**, and the obvious explanation is contradicted by the data.

The finding to explain: within orientation at effect 1.00, replicates that *fail* to reject have a **larger**
mean observed latent angle (66.7°) than those that reject (52.5°). A noise-floor account predicts the
opposite ordering. The same inversion appears in shape at effect 1.00 (83.3° vs 61.5°); the translation null
behaves normally (30.7° rejecting vs 7.8° non-rejecting), so it is specific to cells carrying real signal.

Already ruled out:

- **Latent dimensionality.** Angle rejection is 0.71 at `lv = 2` (n = 7) vs 0.65 at `lv = 3` (n = 93).
- **Weak construction.** The signal survives integration: 81° population → 57° latent, against an 8.9°
  latent null floor.
- **Monte Carlo noise.** A 14-degree gap across a 65/35 split.
- **Cross-talk.** Localization finds nothing projection-associated for orientation's own statistic.

**Leading hypothesis:** the RRPP permutation null for `angle` co-varies with the observed statistic within a
replicate — the statistic is not pivotal, so a replicate whose latent geometry inflates the observed angle
inflates its own null at least as much.

**What to run.** The pilot did not persist null quantiles (`include_null_distributions` was off), so this
cannot be settled from existing records. Re-run a modest set of orientation replicates at effect 1.00 with
`SimulationEvaluationParams(include_null_distributions=True)` and, per replicate, record the observed angle
alongside its null mean and 95th percentile. Then check whether the null quantile tracks the observed
statistic across replicates. If it does, the fix is a pivotal statistic or a studentized test, not a larger
sample size.

**Decision it unblocks:** whether Phase 5 can use the `angle` test as specified, needs a revised statistic,
or needs a revised power target. Nothing else on this list matters if the test itself is mis-calibrated
under signal.

**Status after the sign-anchor fix:** still blocking. `openspec/changes/diagnose-angle-null-pivotality` is
not displaced and proceeds as written.

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

## 4. Choose the Phase 5 design point

Only after items 1–3. The pilot used n = 300 with four stages (75 samples per group-stage cell) over ~660
standardized features.

**What to establish:** how orientation's operating characteristics scale with samples per group-stage cell
and with feature count, so the Phase 5 sample size is chosen on evidence rather than inherited. If item 1
shows the test is mis-calibrated under signal, fix that first — scaling a mis-calibrated test just measures
it more precisely.

**Also decide:** whether the 0.80 orientation power floor is achievable at a defensible sample size, or
whether the target should be revised. Per the Phase 4 exit gate, a failed target means revising the method
or the scientific claim — not merely the Monte Carlo sample size.

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
