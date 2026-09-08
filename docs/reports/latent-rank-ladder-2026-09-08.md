# Phase 5 latent-rank ladder — findings (run 2026-09-08)

**Question (readiness item 3).** At the Phase 5 design point, does retaining more PLS components than the
stage-supervised double CV selects buy orientation `angle` power, and what does it cost magnitude, shape, and
Type I error? Which group-blind rank rule does Phase 5 commit to?

**Verdict: `keep_cv`.** No fixed rank qualifies under the predeclared rule
(`report/rank_decision.json`). Forced 3 is indistinguishable from the CV column; every rank above 3 loses
`shape` power by 0.05–0.43 and gains no `angle` power. **The committed Phase 5 rank rule is stage-supervised
double cross-validation (`plsda_doubleCV`, modal LV across repeats, parsimony tie-break), as in production.**
At the design point it selects rank 3 = `n_stages − 1` (range 2–3 across 1,500 CV units).

Every number below is read from `results/phase5-latent-rank-2026-09-08/report/` (CSV/JSON) or from
`PROVENANCE.txt` in the same directory.

## 1. Configuration and provenance

| Item | Value | Source |
|---|---|---|
| Config | `examples/trajectory_power_study/phase5_latent_rank_ladder.json` | `PROVENANCE.txt` |
| Config sha256 | `b91339f9a8db9ee33d177e9fd9084d5c34e28d5ad2482fc47e4e25232fe31b5f` | `PROVENANCE.txt` |
| Code | `a3ea2b4` (main) + the working tree of OpenSpec change `resolve-latent-rank-at-design-point`, tracked-file diff sha256 `16afc68c…6ed1`, synced to the cluster by rsync | `PROVENANCE.txt` |
| Design point | ρ = 0, `n_samples` = 1200, four stages, `p_dmp` = 0.1, `surgery_censoring` = `error` (default) | config |
| Integration | pooled PLS on M-value methylation; CV knobs `cv1_splits` 3, `cv2_splits` 4, `n_repeats` 5, `max_components` 20, `random_state` 1203 | config |
| Rank axis | `evaluation.integration_params.forced_components` ∈ {`null`, 3, 4, 6, 9, 12}; `null` = CV | config |
| Modes × effects | magnitude, orientation, shape, translation × {0.25, 0.50, 1.00} + one zero-effect anchor per column | config |
| Replicates × permutations | 100 × 199; matched seeds (family `phase5-latent-rank`), shared anchor | config |
| Units | 80 cells × 100 = 8,000; 8,000 present once, 0 signature mismatches, **0 failures**, 0 censored surgeries, 0 duplicated constructions | `PROVENANCE.txt`, `realized_surgery.csv` |
| Compute | SLURM `512x1024`, 100 single-CPU shards, BLAS pinned, no `--n-jobs`; wall 20 min; CV units median 57.4 s (24.1 core-h), forced units median 0.2 s (0.3 core-h) | `PROVENANCE.txt` |
| Versions | python 3.11.16, motco 0.6.0, numpy 2.3.5, scikit-learn 1.8.0, scipy 1.16.3 | `PROVENANCE.txt` |

**Same data, different measurement.** The rank axis is an evaluation-namespace axis, so all six columns share
the primary matched-seed family and identical generator parameters: at every replicate index they evaluate
the same generated dataset at a different retained rank (`rank_decision.json`, `pairing`). Differences
between columns are measurement differences. The rule's pooled independent SE is therefore conservative.

## 2. Decision (`report/rank_decision.json`, `rank_decision.csv`)

Reference = CV column; target orientation/`angle` at e = 1.00; protected magnitude/`delta` and shape/`shape`
at e = 1.00; anchor bound `0.05 + 2·sqrt(0.05·0.95/100) = 0.094`; gain and loss multipliers 2.0.

| Rank | `angle` rate (SE) | (a) gain: diff vs CV, threshold | (b) anchor | (c) protected: `shape` loss, threshold | Qualifies |
|---|---|---|---|---|---|
| CV | 0.85 (0.036) | reference | — | — | — |
| 3 | 0.86 (0.035) | +0.01 ≤ 0.100 ✗ | pass | 0.00 ≤ 0.000 ✓ | no |
| 4 | 0.79 (0.041) | −0.06 ≤ 0.108 ✗ | pass | **0.05 > 0.044** ✗ | no |
| 6 | 0.80 (0.040) | −0.05 ≤ 0.107 ✗ | pass | **0.43 > 0.099** ✗ | no |
| 9 | 0.81 (0.039) | −0.04 ≤ 0.106 ✗ | pass | **0.42 > 0.099** ✗ | no |
| 12 | 0.81 (0.039) | −0.04 ≤ 0.106 ✗ | pass | **0.42 > 0.099** ✗ | no |

Magnitude/`delta` loss is 0.00 at every rank (power 1.00 everywhere). The anchor criterion passes at every
rank (§5). Verdict `keep_cv`; the rationale string in the JSON lists the failing criteria per rank.

## 3. Rate versus rank, per mode (`design_point_operating.csv`, e = 1.00 rows; figure `rank_ladder.png`)

| Mode / statistic | CV | 3 | 4 | 6 | 9 | 12 |
|---|---|---|---|---|---|---|
| orientation / `angle` (target) | 0.85 | 0.86 | 0.79 | 0.80 | 0.81 | 0.81 |
| orientation / `delta` | 0.91 | 0.93 | 0.84 | 0.53 | 0.47 | 0.47 |
| orientation / `shape` | 0.99 | 0.99 | 0.99 | 0.78 | 0.60 | 0.62 |
| magnitude / `delta` (protected) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| magnitude / `angle` | 0.24 | 0.24 | 0.23 | 0.59 | 0.80 | 0.81 |
| magnitude / `shape` | 0.74 | 0.77 | 0.75 | 0.41 | 0.53 | 0.50 |
| shape / `shape` (protected) | 1.00 | 1.00 | 0.95 | 0.57 | 0.58 | 0.58 |
| shape / `angle` | 0.79 | 0.80 | 0.68 | 0.95 | 0.95 | 0.95 |
| shape / `delta` | 0.89 | 0.89 | 0.78 | 0.53 | 0.53 | 0.53 |
| translation / `delta` · `angle` · `shape` | 0.05 · 0.02 · 0.04 | 0.06 · 0.02 · 0.04 | 0.02 · 0.04 · 0.01 | 0.01 · 0.03 · 0.01 | 0.01 · 0.03 · 0.01 | 0.01 · 0.03 · 0.01 |

MC SE is ≤ 0.05 for every entry (0 where the rate is 0 or 1). `component_selection` is `cv` in the CV
column and `forced` in every other column, with `median_selected_lv` equal to the forced rank
(`min_selected_lv` = `max_selected_lv` = rank).

**Does orientation `angle` power move with rank at n = 1200?** No. It is 0.85–0.86 at CV and rank 3 and
0.79–0.81 at every rank from 4 to 12; the differences are within one pooled SE. Across effects
(`design_point_operating.csv`, orientation/`angle`): CV 0.89 / 0.89 / 0.85 at e = 0.25 / 0.50 / 1.00; rank 4
0.75 / 0.76 / 0.79; ranks 6–12 0.86–0.87 / 0.81–0.83 / 0.80–0.81. The stage-supervised rank retains
everything the `angle` test can use at this design point; components beyond `n_stages − 1` add nothing to
the target.

**What a higher rank costs.** Above rank 3 the latent space admits directions the stage label does not
organize. `shape` power falls in every mode that has it — orientation 0.99 → 0.60–0.78, shape 1.00 → 0.57,
magnitude 0.74 → 0.41–0.53 — and `delta` falls in the orientation and shape modes (0.91 → 0.47, 0.89 →
0.53). Only magnitude/`delta` is rank-invariant at 1.00. At the same time the **off-target `angle` response
rises**: magnitude→`angle` 0.24 → 0.81 and shape→`angle` 0.79 → 0.95 by rank 9. A fixed rank above
`n_stages − 1` therefore degrades specificity in both directions: the geometry each mode targets is
diluted, and PC1 of the stage configuration is rotated by group-specific noise directions the extra
components carry.

**Forced 3 versus CV.** Every forced-3 entry is within 0.03 of the CV column (the largest difference is
orientation/`delta` 0.93 vs 0.91). CV selection noise (range 2–3) is immaterial at this design point, so
"fixed `n_stages − 1`" would be an operationally equivalent group-blind rule; it offers no gain, and CV
remains the rule that adapts if the stage structure changes.

## 4. Covariates: eigengap and `angle` null width (`design_point_operating.csv`)

| Row | | CV | 3 | 4 | 6 | 9 | 12 |
|---|---|---|---|---|---|---|---|
| anchor (`none`, e = 0) | median eigengap | 0.048 | 0.048 | 0.047 | 0.047 | 0.047 | 0.047 |
| | median `angle` null q95 (°) | 5.9 | 6.0 | 10.8 | 10.9 | 11.1 | 11.2 |
| | IQR `angle` null q95 (°) | 4.9 | 4.9 | 5.1 | 5.2 | 5.1 | 5.1 |
| orientation e = 1.00 | median eigengap | 0.045 | 0.045 | 0.040 | 0.040 | 0.040 | 0.040 |
| | median `angle` null q95 (°) | 11.5 | 10.2 | 17.0 | 15.6 | 16.2 | 16.4 |
| | IQR `angle` null q95 (°) | 14.5 | 10.8 | 30.2 | 22.3 | 22.2 | 23.7 |

The recorded pooled eigengap is nearly constant across rank on identical data (0.048 → 0.047 at the anchor,
0.045 → 0.040 under orientation), the sanity check design D8 asked for: the ladder changes the measurement
space, not the baseline geometry, and the small drop reflects the extra near-zero eigenvalues a larger
configuration carries. The `angle` null, by contrast, widens as soon as rank exceeds `n_stages − 1`: the
anchor's median q95 nearly doubles (5.9° → 10.8°) and the orientation column's dispersion doubles (IQR 14.5°
→ 30.2° at rank 4). A wider, more variable null with an unchanged eigengap is the signature of the extra
components adding noise directions rather than signal, consistent with the pivotality finding that the
null tracks the replicate's own latent configuration.

## 5. Anchors and Type I error

Per-column zero-effect anchors (`design_point_operating.csv`, `none` at e = 0, 100 replicates each):

| Statistic | CV | 3 | 4 | 6 | 9 | 12 | Bound |
|---|---|---|---|---|---|---|---|
| `delta` | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.094 |
| `angle` | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.094 |
| `shape` | 0.03 | 0.03 | 0.00 | 0.00 | 0.00 | 0.00 | 0.094 |

No rank inflates any anchor. At ranks ≥ 4 `delta` and `shape` reject 0 of 100 — a fixed rank above
`n_stages − 1` makes those tests *conservative* under the null, which the one-sided anchor bound does not
penalize but which is one more cost of over-retention.

Independent-seed Type I controls at the baseline (`type_i_table.csv`, `acceptance_report.csv`, CV rank):
`none` 0.06 / 0.05 / 0.07 (`delta` / `angle` / `shape`, combined 0.14); `translation` at e = 1.00
0.08 / 0.07 / 0.09 (combined 0.17). Every `type_i_control` target is met within 2 SE
(`acceptance_report.json`).

## 6. Re-measurement at the chosen design point under CV (`design_point_operating.csv`, CV column)

The design-point pilot ran only orientation and translation; `p_dmp` had changed from 0.2 to 0.1. The CV
column re-measures the other modes at the committed configuration:

| Mode / statistic | e = 0.25 | 0.50 | 1.00 |
|---|---|---|---|
| magnitude / `delta` | 1.00 | 1.00 | 1.00 |
| shape / `shape` | 0.95 | 0.97 | 1.00 |
| orientation / `angle` | 0.89 | 0.89 | 0.85 |
| orientation / `shape` (cross-talk) | 1.00 | 1.00 | 0.99 |
| translation / `angle` | 0.02 | 0.02 | 0.02 |

Magnitude and shape hold power ≥ 0.95 at every effect with `p_dmp = 0.1`. Orientation `angle` is flat
across effects (0.85–0.89; differences within MC error).

**Agreement with the design-point pilot.** The pilot's ρ = 0, n = 1200 column gave orientation/`angle`
0.88 (SE 0.032) at e = 1.00 with median selected rank 3 and median eigengap 0.049
(`results/phase5-design-point-2026-09-08/report/design_point_operating.csv`); the ladder's CV column gives
0.85 (SE 0.036), rank 3, eigengap 0.045. The difference (0.03) is well within the pooled SE (0.048); the two
runs draw different matched-seed families, so this is an independent replication of the design-point
result.

**Orientation → shape at the design point.** At CV rank the projection-associated orientation→`shape`
response is 0.99–1.00 at every effect — it does **not** decay at the committed rank. It decays only as the
rank grows (0.99 → 0.78 → 0.60 → 0.62 at ranks 4 → 6 → 9 → 12 at e = 1.00; 0.98 → 0.96 at e = 0.25), as the
2026-09-03 probe found at the old design point, now confirmed on the corrected estimator at ρ = 0,
n = 1200, `p_dmp = 0.1`. Since the rank is not being raised, Phase 5 keeps the item-2 predeclaration:
orientation's `shape` response is reported as projection-associated cross-talk, not as evidence about the
shape estimator.

## 7. Committed rule and limitations

**Committed Phase 5 rank rule (group-blind):** stage-supervised double cross-validation
(`plsda_doubleCV`, modal LV across repeats with parsimony tie-break) selects the retained PLS rank; no group
label enters the sizing. At the Phase 5 design point this yields rank 3 = `n_stages − 1`. The rule is
committed for the design point Phase 5 runs; a fixed rank on real data would be a preregistered choice,
not a tuned one, and this ladder gives no reason to make it.

Limitations:

- One design point (ρ = 0, n = 1200, `p_dmp = 0.1`, four stages). The rank response could differ with more
  stages (where `n_stages − 1` is larger) or a trending baseline (ρ = 0.8 moved the CV to rank 7–8 in the
  pilot).
- 100 replicates: MC SE up to 0.05 per rate; the gain criterion could not detect an `angle` improvement
  smaller than ≈ 0.10. The observed differences are negative, so this does not change the verdict.
- The rule's pooled SE treats paired columns as independent (conservative by design).
- The anchor bound is one-sided (upper); the conservative `delta`/`shape` anchors at ranks ≥ 4 are reported,
  not penalized.
- The eigengap is a property of the configuration in each column's latent space; its small drop with rank is
  a measurement-space effect, not a baseline-geometry change.
- The code revision in `PROVENANCE.txt` is the pre-change commit plus the change's working tree (diff
  sha256 recorded); the merge commit of the change is the reproducible revision.

## 8. Reproduction

```bash
# Cluster (as executed 2026-09-08; partition/resource flags are cluster-specific).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN=results/phase5-latent-rank-2026-09-08
sbatch -p 512x1024 --cpus-per-task=1 --mem=2G --time=6:00:00 --array=0-99 \
    --export=ALL,STUDY_CONFIG=$(pwd)/examples/trajectory_power_study/phase5_latent_rank_ladder.json,STUDY_OUT=$(pwd)/$RUN,N_SHARDS=100 \
    scripts/motco_study_array.sbatch          # job 878624
python scripts/motco_study.py merge  --out-dir $RUN
python scripts/motco_study.py report \
    --config examples/trajectory_power_study/phase5_latent_rank_ladder.json --out-dir $RUN
```

Outputs under `report/`: `design_point_operating.csv` (with `component_selection`), `rank_ladder.png`,
`rank_decision.json` / `.csv`, `realized_surgery.csv`, `type_i_table.csv`, `acceptance_report.{csv,json}`,
`config_spectrum.csv`, `eigengap_stratified_power.csv`, `power_curves.{csv,png}`,
`specificity_matrix.{csv,png}`, `type_i.png`, `design_point_power.png`.
