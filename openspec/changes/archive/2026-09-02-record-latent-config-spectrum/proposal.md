# record-latent-config-spectrum

## Why

The [geometry audit](../../../docs/reports/geometry-audit-2026-09-01.md) (finding F1) closed the gap the pivotality diagnostic left open. The pivotality report proved that each orientation replicate's outcome is decided by the width of its own `angle` permutation null — per-replicate critical values span 5.0°–176.6° — and stated that *"nothing the harness records today identifies which replicates will be resolvable"*: selected dimensionality and CV AUROC both saturate and neither correlates with the null width. The audit found the missing quantity: the **relative eigengap (λ₁−λ₂)/Σλ of the centered stage-mean configuration** in the latent space. Regenerating pivotality replicates bit-for-bit from persisted seeds, the pooled-configuration eigengap predicts the recorded `angle` null q95 with Spearman −0.75 across all 100 orientation replicates (−0.81 within the 16 extremes); the 8 narrowest-null replicates average gap 0.097 against 0.046 for the 8 widest, a 2× eigengap separation mapping to a 26× null-width separation; replicates that reject average gap 0.053 against 0.035 for those that fail.

Three open readiness items ask for exactly this quantity. Item 1's left-open gap is the resolvability qualifier itself. Item 3 needs "a direct measure of latent trajectory-geometry stability, which does not exist yet" before latent dimensionality can be reconsidered on evidence. Item 4's design-point study must measure how null-width dispersion contracts with sample size, stratified by the geometry that governs it. The cost is one SVD of a k×d matrix per configuration — negligible beside `estimate_difference`. Per the audit's sequencing, this change (with P2) lands **before any new power runs**, so every future record carries the covariate.

## What Changes

- Evaluation results gain a JSON-safe **configuration-spectrum block** beside `null_summary`: for the pooled and each per-group centered stage-mean configuration in the evaluated latent space, the normalized eigenvalue spectrum and the relative eigengap, computed from the same fitted LS-mean vectors the trajectory statistics already use.
- Whenever RRPP runs, the **per-permutation pooled eigengap** is summarized (retained count, mean, sd, quantiles) so a replicate can locate its observed eigengap against its own permutation distribution. The spectrum rides an opt-in path through the estimator; permutation draws, observed statistics, and p-values remain byte-identical to the pre-change pipeline.
- Grid replicate records persist the spectrum block, and the **cell parameter signature gains an explicit spectrum schema version**. Deliberately unlike the null-summary change (which was signature-neutral): a pre-change shard refuses to resume rather than producing a mixed result set in which some records silently lack the field the Phase-5 stratified tables require. Records already written still load (the block defaults to empty).
- The study report gains an **eigengap-stratified rejection-rate table** for orientation-mode cells, and the realized eigengap joins the per-cell geometry summaries.
- The pivotality analysis gains the **eigengap covariate**: per cell and statistic, the association between the recorded eigengap and the replicate's own null width, beside the existing observed-vs-null associations. Acceptance: re-running the analysis over records generated under the new schema reproduces the audit's association (negative, and materially strong for `angle` in the orientation cell).
- **Non-goals:** no change to any statistic, test decision, permutation draw, generator behavior, integration method, or latent-space sizing; full per-permutation spectra are not retained; no baseline-continuity axis (that is plan item P4) and no effect-axis policy change (P2).

## Capabilities

### Modified Capabilities

- `simulation-evaluation-harness`: Evaluation results carry the latent stage-mean configuration spectrum (pooled and per group) on every run, and a per-permutation eigengap summary whenever permutations are run — additively and without altering any statistic or p-value.
- `simulation-grid-orchestration`: Persisted replicate records carry the spectrum block; the parameter signature versions the spectrum schema so resume cannot mix pre- and post-change contracts; legacy records still load.
- `trajectory-power-study`: Study reporting summarizes realized eigengaps per cell and stratifies orientation power by the recorded eigengap.
- `angle-null-pivotality-diagnostic`: The pivotality analysis reports the association between the recorded configuration eigengap and each replicate's own null width, closing the report's "invisible to every persisted diagnostic" gap.

## Impact

- `src/motco/stats/trajectory.py` — opt-in computation of the stage-mean configuration spectra from the LS-mean vectors already fitted inside `estimate_difference`; the default return contract is unchanged. `src/motco/stats/permutation.py` — opt-in forwarding through `RRPP` (serial and parallel paths), consuming no RNG.
- `src/motco/simulations/evaluation.py` — the spectrum block on `SimulationEvaluationResult`; `src/motco/simulations/grid.py` — record field, serialization, and the signature version key; `src/motco/simulations/study/report.py` / `summary.py` — the stratified table; `src/motco/simulations/pivotality.py` — the eigengap covariate.
- Tests beside each surface, including byte-identical regression checks on statistics, p-values, and permutation draws, and a mirror of the audit's eigengap definition against its reproduction snippet.
- Docs: `simulations/study/README.md` (record field + report table), `docs/phase5-readiness.md` (items 1/3/4 pointers to the now-recorded measure).
- Existing merged results (`results/phase4-2026-08-27/`, `results/angle-pivotality-2026-09-01/`) remain loadable and their committed reports untouched; existing shards can no longer be resumed into — intended, per the audit.
- No new runtime dependency; no R.
