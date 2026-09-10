# Phase 5 paper-grade trajectory power study — 2026-09-10

**Run:** `results/phase5-2026-09-10/` · **Config:** `examples/trajectory_power_study/phase5_power_study.json`
(sha256 `4f10aa1f…09826`, from [`PROVENANCE.txt`](../../results/phase5-2026-09-10/PROVENANCE.txt)) ·
**Gate decision: HOLD** (from `report/phase4_gate_decision.json`).

The HOLD is carried by the two mandatory-control rules on the magnitude mode — magnitude/`angle` 0.276 and
magnitude/`shape` 0.742 against a 0.0695 bound — not by any power rule. All three mandatory power diagonals
met their 0.80 floor, every Type I check met, and the record set is complete. Both control failures reproduce
values the latent-rank ladder pilot had already measured (0.24 and 0.74), so paper-grade precision did not
discover them; it confirmed them at SE ≈ 0.020 and removed Monte Carlo error as an explanation.

## 1. Configuration and provenance

The design point is fixed, each coordinate traced to the readiness item that chose it:

| coordinate | value | chosen by |
|---|---|---|
| baseline continuity ρ | 0 (independent baseline, isotropic endpoint) | [design-point pilot](phase5-design-point-pilot-2026-09-08.md), readiness item 4 |
| n | 1200 (300 per group-stage cell) | same |
| stages / `p_dmp` | 4 / 0.1 | same |
| measurement space | pooled PLS on M-value methylation | readiness items 1–2 |
| retained rank | stage-supervised double CV (`cv1_splits` 3, `cv2_splits` 4, 5 repeats, ≤ 20 components) | [latent-rank ladder](latent-rank-ladder-2026-09-08.md) returned `keep_cv`, readiness item 3 |

Monte Carlo sizing is 500 replicates × 999 permutations per unit, 19 cells, 9,500 units. The matched-seed
family is `phase5-primary`, new to this study, so the run is independent of both Phase 5 pilots; the
configuration otherwise derives from the ladder's cross-validated column (`metadata.derives_from`) with
`generator` and `evaluation.integration_params` copied verbatim, no `design_grid`, and permutations raised
from 199 to 999.

The double-CV rank rule selected **rank 3 = `n_stages − 1`** at every cell (`report/phase4_pls_selection.csv`:
median and modal `selected_lv` 3 in all 19 cells, mean 2.90–3.18, range 2–4; `cv_settings_consistent` true
throughout). Rank selection is therefore not a source of between-cell variation in this run.

[`PROVENANCE.txt`](../../results/phase5-2026-09-10/PROVENANCE.txt) carries the full field list: code revision
`b8f7daa` on a clean tree, the exact `sbatch` line with `STUDY_N_JOBS` unset, BLAS pinned to one thread,
job 880599, both environments' package versions, the wall interval, per-unit timings, and the merge/report
split. Two provenance points worth reading directly:

- **Cost.** 182.7 recorded core-hours against the config's ~150 EPYC core-hour budget, a 22 % overage. The
  budget was projected from the ladder's 57 s median per unit; the realized median is 61.5 s with a
  right-skewed tail (mean 69.2 s, max 191.8 s) because 68 of the 100 array tasks shared node `n3`. Attribution
  is not the cause — the 2,000 attribution-bearing orientation units ran at a 60.0 s median against 62.1 s for
  everything else, matching the rehearsal's 42.9 s vs 43.0 s. Nothing about the measurement changed.
- **Split environments.** Records were produced on EPYC 7662 nodes (python 3.11.16, uv 0.12.10); the report
  was rendered on the workstation (python 3.11.15, uv 0.11.17) from the rsynced `merged.jsonl`. Reporting
  re-fits nothing, so this is deterministic post-processing of recorded statistics.

## 2. Unit and failure accounting

Every completeness assertion below is read from the merged records, not inferred from SLURM exit codes.

| check | result |
|---|---|
| units expected / present exactly once | 9,500 / 9,500; 0 duplicate `(cell, replicate)` pairs |
| enumerated cells present | 19 / 19, exactly 500 replicates each, no unexpected cell |
| parameter-signature mismatches | 0 |
| `status = failed` records | 0 (all 9,500 `completed`); no `diagnostic_error_type` to report |
| censored surgeries | 0 at every cell (`report/realized_surgery.csv`: `censored_fraction` 0.0, `realized_mean` = `nominal_size`) |
| attribution accounting | 2,000 eligible / 2,000 computed / 0 failed (`report/driver_report.csv`) |
| `n_jobs` uniformity | 1 in all 9,500 records (`report/report_contract.json`) |
| permutations | 999 in all 9,500 records |
| array tasks | 100 / 100 `COMPLETED`, exit `0:0`, empty stderr |

The gate's own completeness rule agrees (`report/phase4_gate_decision.json`, `diagnostic_completeness`: met,
9,500 records, 0 failed, 9,500 PLS records, 0 missing integration metadata, 0 missing realized geometry,
2,000 attribution eligible, 0 incomplete). No unit was dropped, and none needed to be — task 2.3's
resubmission path was never exercised.

Realized surgery sizes sit exactly on nominal at every pool-limited cell, e.g. orientation at e = 1.00
nominal 125.662 / realized mean 125.662 (range 97–158) and translation at e = 1.00 nominal 37 / realized 37
(range 37–37). The magnitude cells and the `none` cells report `surgery_replicates = 0`, which is correct
rather than missing: magnitude scales δ globally and `none` is the identity, so neither draws from the finite
CpG pool.

## 3. Gate decision

**HOLD.** Rationale as recorded: *"Mandatory gate(s) failed: mandatory_control[magnitude,angle]
(rate=0.2760 > bound=0.0695 (alpha=0.05 + 2·SE, n=500)); mandatory_control[magnitude,shape] (rate=0.7420 >
bound=0.0695 (alpha=0.05 + 2·SE, n=500))"* — `report/phase4_gate_decision.json`. No confirmation re-run is
required by the decision (`confirmation_runs: []`).

Per-rule outcomes (`report/phase4_gate.csv`, `report/phase4_gate_decision.json`):

| rule | kind | outcome | observation |
|---|---|---|---|
| `mandatory_power[magnitude,delta]` | power | **met** | top_rate 1.000 ≥ 0.800, monotone |
| `mandatory_power[orientation,angle]` | power | **met** | top_rate 0.840 ≥ 0.800; `tolerated_monotone` true |
| `mandatory_power[shape,shape]` | power | **met** | top_rate 0.996 ≥ 0.800, monotone |
| `mandatory_control[magnitude,angle]` | control | **failed** | 0.2760 > 0.0695, exceedance 10.3 SE |
| `mandatory_control[magnitude,shape]` | control | **failed** | 0.7420 > 0.0695, exceedance 34.4 SE |
| `descriptive[orientation,delta]` | descriptive | not gated | 0.036 → 0.804 → 0.878 → 0.894 → 0.912 |
| `descriptive[orientation,shape]` | descriptive | not gated | 0.048 → 1.000 → 1.000 → 1.000 → 0.998 |
| `descriptive[shape,delta]` | descriptive | not gated | 0.036 → 0.806 → 0.874 → 0.934 → 0.954 |
| `descriptive[shape,angle]` | descriptive | not gated | 0.046 → 0.692 → 0.816 → 0.800 → 0.796 |
| `type_i_inflation[…]` × 21 | Type I | **all met** | every control cell ≤ 0.0695 |
| `diagnostic_completeness` | completeness | **met** | see §2 |

**The two predeclared readers disagree on orientation/`angle`, and the disagreement is about monotonicity,
not about the floor.** The Phase 4 gate records `mandatory_power[orientation,angle]` as met — the floor is
cleared and non-monotonicity is tolerated within 2·SE. The acceptance-target reader records
`power[orientation,angle]` as **not met** (`report/acceptance_report.csv`, row 7:
*"monotone=False; top_rate=0.840 >= floor=0.800"*), because the curve declines from 0.878 at e = 0.50 to
0.840 at e = 1.00 and its strict monotonicity rule admits no tolerance. Both are predeclared; neither is
overridden here. The substantive reading is in §5: the orientation curve is flat, not rising, above
e = 0.25.

Acceptance targets overall (`report/acceptance_report.csv`): 6 / 6 `type_i_control` met, 2 / 3
`power_monotonicity` met (orientation/`angle` the exception above), 3 / 5 `specificity` met (translation × 3
met; magnitude/`angle` and magnitude/`shape` failed, the same two rules that carry the gate's HOLD).

## 4. Type I error

The two `type_i_baseline` cells carry the Type I acceptance target and are a separate seed family from the
power grid (`report/type_i_table.csv`):

| cell | mode | `delta` | `angle` | `shape` | combined rule |
|---|---|---|---|---|---|
| `type_i_baseline-c9b8da02420a` | `none` | 0.050 (SE .0097) | 0.040 (SE .0088) | 0.054 (SE .0101) | 0.126 (SE .0148) |
| `type_i_baseline-d18f474778c9` | translation, e = 1.00 | 0.044 (SE .0092) | 0.048 (SE .0096) | 0.062 (SE .0108) | 0.132 (SE .0151) |

All six per-statistic rates are within 2·SE of α = 0.05 (`report/acceptance_report.csv`, rows 0–5; largest
deviation 0.012 against a 0.0216 tolerance). The combined-rule rates of 0.126 and 0.132 are consistent with
the ≈ 0.143 expected from three α = 0.05 tests without multiplicity control, and are reported as a property
of the combined rule, not as inflation of any single statistic.

**The shared zero-effect anchor is one measurement, not four.** Cell `power_primary-d74045b65506`
(`trajectory_mode` `none`, e = 0.00) supplies the 0.00 power point for all four modes — `resolves_modes`
`[magnitude, orientation, shape, translation]`, `counted_as: 1` in `report/report_contract.json`. Its rates
are `delta` 0.036 (SE .0083), `angle` 0.046 (SE .0094), `shape` 0.048 (SE .0096), all nominal. It is counted
once wherever it appears below.

**Translation at e > 0 is a negative control on all three statistics** and behaves as one across the whole
sweep (`report/power_curves.csv`): `delta` 0.054 / 0.056 / 0.040 / 0.042, `angle` 0.038 / 0.028 / 0.050 /
0.042, `shape` 0.046 / 0.046 / 0.044 / 0.030 at e = 0.25 / 0.50 / 0.75 / 1.00. A constant observed-space
location offset moves none of the three statistics, which is the invariance the estimators claim.

## 5. Power per mode

Diagonal power, from `report/power_curves.csv` (figure `report/power_curves.png`); e = 0.00 is the shared
anchor in every row:

| mode / statistic | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 | SE at 1.00 |
|---|---|---|---|---|---|---|
| magnitude / `delta` | 0.036 | 1.000 | 1.000 | 1.000 | **1.000** | 0.000 |
| orientation / `angle` | 0.046 | 0.842 | 0.878 | 0.872 | **0.840** | 0.016 |
| shape / `shape` | 0.048 | 0.954 | 0.988 | 0.994 | **0.996** | 0.003 |

Magnitude/`delta` and shape/`shape` clear the 0.80 floor with room to spare and rise monotonically.
Orientation/`angle` clears the floor at 0.840 but is **flat from e = 0.25 onward** (0.842, 0.878, 0.872,
0.840) — it saturates immediately and then drifts down, which is what fails the strict acceptance rule in §3.

### Orientation read beside its covariates

Every orientation number is reported with the recorded eigengap and `angle` null width, because those are the
observables that transfer to real data.

Eigengap (`report/config_spectrum.csv`, median with q25–q75, per configuration and pooled):

| effect | A | B | pooled |
|---|---|---|---|
| anchor (e = 0.00) | 0.0533 (.0357–.0801) | 0.0521 (.0360–.0789) | 0.0530 (.0351–.0798) |
| 0.25 | 0.0575 | 0.0575 | 0.0534 (.0368–.0759) |
| 0.50 | 0.0623 | 0.0610 | 0.0539 (.0325–.0779) |
| 0.75 | 0.0657 | 0.0609 | 0.0523 (.0314–.0728) |
| 1.00 | 0.0683 | 0.0660 | 0.0487 (.0318–.0723) |

`angle` null width, `null_summary["angle"]["q95"]` per replicate, computed from `merged.jsonl` (degrees):

| cell | median q95 | IQR |
|---|---|---|
| anchor (`none`, e = 0.00) | 5.457 | 4.933 |
| orientation e = 0.25 | 6.501 | 5.557 |
| orientation e = 0.50 | 7.227 | 7.519 |
| orientation e = 0.75 | 8.623 | 9.428 |
| orientation e = 1.00 | 11.286 | 13.919 |

The null widens with the effect it is meant to test — from 5.5° at the anchor to 11.3° at e = 1.00, with the
IQR growing faster than the median (4.9° → 13.9°). This is the pivotality behaviour already established in the
[`angle`-null pivotality report](angle-null-pivotality-2026-09-01.md) (the null tracks its own observed
statistic, slope 0.811), and it is the mechanism behind the flat orientation curve: a larger nominal
orientation surgery buys a larger observed angle *and* a proportionally larger critical value, so the
rejection rate stops improving.

Eigengap-stratified orientation/`angle` power (`report/eigengap_stratified_power.csv`, terciles):

| effect | low tercile | middle | high |
|---|---|---|---|
| 0.25 | 0.778 (mean gap .0279) | 0.862 (.0541) | 0.886 (.0956) |
| 0.50 | 0.820 (.0245) | 0.880 (.0543) | 0.934 (.0963) |
| 0.75 | 0.784 (.0245) | 0.922 (.0521) | 0.910 (.0917) |
| 1.00 | **0.754** (.0245) | 0.886 (.0499) | 0.880 (.0889) |

**Power is eigengap-dependent, and in the lowest tercile at e = 1.00 it falls below the 0.80 floor (0.754,
SE 0.033).** The pooled 0.840 is an average over a spread that straddles the floor. Any real-data orientation
result must therefore be read against its own recorded eigengap, not against the pooled figure — this is the
concrete form of the n-conditional limitation in §8.

### Cross-check against the pilots (design decision D6)

Performed before any prose was written. All four required comparisons agree within Monte Carlo error:

| quantity | paper-grade (500×999) | ladder CV column (100×199) | design-point pilot (100×199) | verdict |
|---|---|---|---|---|
| orientation / `angle` @ e = 1.00 | 0.840 ± 0.016 | 0.85 ± 0.036 | 0.88 ± 0.032 | agrees (≤ 1.1 pooled SE) |
| magnitude / `delta` @ e = 1.00 | 1.000 | 1.00 | — | agrees |
| shape / `shape` @ e = 1.00 | 0.996 ± 0.003 | 1.00 | — | agrees |
| anchor `delta` / `angle` / `shape` | 0.036 / 0.046 / 0.048 | 0.01 / 0.02 / 0.03 | 0.04 / 0.04 / 0.03 | agrees, all nominal |

The two failing controls also reproduce: magnitude/`angle` 0.276 here vs 0.24 in the ladder, magnitude/`shape`
0.742 here vs 0.74. No material discrepancy was found, so nothing required investigation before writing. The
practical consequence is stated at the top of this report: the HOLD is a reproduced, precision-confirmed
result, and the 22-fold increase in Monte Carlo work did not change any conclusion the pilots supported.

## 6. Cross-talk

Off-diagonal rejection rates at e = 1.00 (`report/specificity_matrix.csv`, figure
`report/specificity_matrix.png`; rows are the mode applied, columns the statistic tested):

| mode ↓ / statistic → | `delta` | `angle` | `shape` |
|---|---|---|---|
| magnitude | **1.000** | 0.276 | 0.742 |
| orientation | 0.912 | **0.840** | 0.998 |
| shape | 0.954 | 0.796 | **0.996** |
| translation | 0.042 | 0.042 | 0.030 |
| `none` (anchor) | 0.036 | 0.046 | 0.048 |

**Magnitude's off-diagonals are mandatory controls and both fail** the α + 2·SE bound of 0.0695:
magnitude/`angle` 0.276 (exceedance 10.3 SE) and magnitude/`shape` 0.742 (34.4 SE). The localization table
(`report/phase4_localization.csv`) splits them:

- magnitude/`angle` is **`construction_present`**, first material at the `population_standardized`
  checkpoint, normalized excess rising 0.064 → 0.110 → 0.148 → 0.176 across effects. The δ-scaling surgery
  genuinely reorients the trajectory in standardized feature space before integration, so the `angle` test is
  detecting a real difference that the construction was not intended to create. This is a construction
  impurity, not evidence against the orientation estimator.
- magnitude/`shape` is classified **`not_material` at every effect** — no checkpoint's normalized shape
  difference exceeded the 0.05 materiality threshold — yet the `shape` test rejects 74.2 % of the time. The
  two statements are reconcilable, and the records say how: at magnitude e = 1.00 the observed `shape`
  statistic has median 0.0233 against a permutation null whose q95 has median 0.0181 (IQR 0.0015, n = 500,
  computed from `merged.jsonl`; the anchor's null q95 median is 0.0095). The observed difference is real and
  sits above its own critical value — hence 74.2 % rejection — while being an order of magnitude below a
  0.05 *normalized* threshold. **The materiality threshold is not calibrated to the `shape` statistic's
  scale**, so `not_material` here means "small in normalized units", not "absent". Whether the underlying
  response is a genuine `shape` signature of pure δ-scaling or an artefact of how the surgery realizes size
  changes is the substantive question the HOLD hands forward; it is not resolved here.

**Orientation → `shape` (0.998) is a descriptive gate role, not an acceptance target,** and was predeclared
as projection-associated cross-talk of the rank-3 stage-supervised latent space (readiness item 2;
[latent-rank ladder](latent-rank-ladder-2026-09-08.md) §6, 0.99–1.00 at CV rank). This run reproduces the
magnitude of the response and its predeclared role stands unchanged.

**One divergence from the predeclared label must be recorded.** This run's localization classifies
orientation/`shape` as **`construction_present`** at the `population_standardized` checkpoint for e ≥ 0.50
(normalized excess 0.052 / 0.057 / 0.059), not as projection-associated; the pair that comes back
`projection_associated` at the `pls_latent` checkpoint is orientation/**`delta`** at e = 1.00 (normalized
value 0.0601 against a null of 0.0026). The predeclared framing therefore attached the
"projection-associated" label to the wrong off-diagonal for this run's data. Because both orientation
off-diagonals are descriptive roles and neither gates anything, this changes no verdict — but the paper must
not repeat the projection-associated claim for orientation/`shape` on the strength of this run, and the exit
review should reconcile the label with the localization instrument.

**Shape's off-diagonals are construction-present descriptives.** shape/`angle` 0.796 is
`construction_present` with a large normalized excess (0.230 → 0.343 → 0.406 → 0.451), and shape/`delta`
0.954 becomes `construction_present` at e = 1.00 (0.0525) — the interior-stage permutation moves path length
and orientation in feature space, as the geometry audit anticipated.

## 7. Drivers

Restricted to the declared component — `observed` — from `report/driver_report.csv`
(`report_contract.json`: `driver_component: observed`, because the Phase 4 pilot measured `pls_captured`
precision at 0.15 against `observed` 1.00; `phase4_attribution.csv` retains all three components).
All 2,000 eligible replicates computed, 0 failed.

| effect | transition | precision | recall | mean selected | bootstrap sign stability | bootstrap top-k frequency |
|---|---|---|---|---|---|---|
| 0.25 | 0→1 / 1→2 / 2→3 | 1.000 | 0.262 / 0.255 / 0.260 | 20.0 | 0.788 / 0.793 / 0.790 | 0.030 |
| 0.50 | 0→1 / 1→2 / 2→3 | 1.000 | 0.134 / 0.134 / 0.134 | 20.0 | 0.817 / 0.818 / 0.821 | 0.030 |
| 0.75 | 0→1 / 1→2 / 2→3 | 1.000 | 0.094 / 0.093 / 0.093 | 20.0 | 0.839 / 0.839 / 0.838 | 0.030 |
| 1.00 | 0→1 / 1→2 / 2→3 | 1.000 | 0.073 / 0.072 / 0.073 | 20.0 | 0.856 / 0.859 / 0.857 | 0.030 |

Precision is 1.000 at every effect and transition: every feature the top-20 selection names is a genuine
differential site. Recall falls from 0.26 to 0.07 as the effect grows, which is arithmetic rather than
degradation — the selection is capped at `top_k = 20` while the true driver set grows with the surgery size
(nominal 31 sites at e = 0.25 up to 126 at e = 1.00, `report/realized_surgery.csv`), so recall is bounded
above by 20 / |truth|. Within-replicate bootstrap sign stability improves with effect (0.79 → 0.86), and the
three transitions agree closely at every effect, as they should for a single global per-omic permutation.

> **No cross-replicate driver-stability claim is made.** Cross-replicate top-k Jaccard and sign agreement
> (`top_k_jaccard`, `sign_agreement` in `report/phase4_attribution.csv`) are `descriptive` only: every
> replicate index draws a fresh differential-indicator set, so the true driver set genuinely differs across
> replicates and matched seeds pair cells *within* a replicate index, not across them. A stability claim needs
> a design that holds the driver set fixed, which this study is not.

Verified in the outputs: `driver_report.csv` carries no `top_k_jaccard` or `sign_agreement` column, and the
attribution figure `report/phase4_attribution_stability.png` is titled "Within-replicate bootstrap stability
(observed component)" and drawn from the bootstrap series only.

## 8. Construction limitations

- **The orientation power claim is n-conditional.** It holds at n = 1200 (300 per group-stage cell) at the
  isotropic endpoint ρ = 0. Along n at fixed ρ the eigengap is constant while the `angle` null width
  contracts, so the recorded eigengap distribution — not n — is the observable that carries the claim to real
  data. This run makes that concrete: pooled orientation/`angle` power at e = 1.00 is 0.840, but stratified by
  eigengap tercile it is 0.754 / 0.886 / 0.880, so the lowest tercile sits below the predeclared 0.80 floor
  (`report/eigengap_stratified_power.csv`). A cohort whose recorded eigengap lands in that tercile should not
  be assumed to inherit the pooled figure.
- **The orientation surgery's realized latent contrast is not ρ-invariant.** The same nominal effect realizes
  a smaller contrast on a trending baseline (design-point pilot: median observed angle 28° at ρ = 0.5 vs 66°
  at ρ = 0), so orientation power at another continuity is a different construction. Baseline continuity must
  not be read as a lever that buys power. This run sweeps no continuity axis — the absence of
  `report/continuity_resolved_orientation.csv` is the report's statement that ρ was held at 0.
- **The eigengap, not ρ, transfers to real data.** `delta`/`angle`/`shape` are measured *within* the
  constructed PLS latent space; that space is the measurement substrate, and the viz down-projection is
  display-only.
- **The flat orientation curve is a property of the `angle` null, not a power ceiling to be tuned away.**
  Rejection saturates at ≈ 0.84–0.88 from e = 0.25 because the null width grows with the observed statistic
  (§5). No replicate-count or rank change addresses it; a different pivot would.

### Deviations from the predeclared targets, and the revision each implies

| deviation | revision implied |
|---|---|
| `mandatory_control[magnitude,angle]` 0.276 > 0.0695 | **Method or claim revision.** magnitude/`angle` is `construction_present` in standardized feature space, so either the magnitude surgery is corrected to be orientation-pure, or the paper drops the claim that `angle` is specific against a pure size change and reports the response as construction impurity. |
| `mandatory_control[magnitude,shape]` 0.742 > 0.0695 | **Method revision, with a diagnostic prerequisite.** The records show a real but numerically small `shape` response (observed median 0.0233 vs null q95 median 0.0181) that the 0.05 normalized materiality threshold reports as `not_material`, so the threshold must be recalibrated to the statistic's scale before the localization instrument can classify this pair at all. Until that is settled the specificity of `shape` against pure δ-scaling cannot be claimed. |
| `power[orientation,angle]` acceptance target not met (monotonicity) | **Claim revision.** The floor is cleared; the strictly-monotone form of the claim is not supported and should be replaced by a saturating-response claim stated with the eigengap stratification. |
| orientation/`shape` labelled projection-associated, localized as `construction_present` | **Claim revision, no gate impact.** Both orientation off-diagonals are descriptive; the label must be reconciled with `phase4_localization.csv` before the paper repeats it. |

None of these is a Monte Carlo sample-size question, and none is addressed by raising the replicate count —
which the Phase 4 exit gate forbids in any case. Consistent with the change's scope, no remedy is implemented
here; the HOLD and these four revisions are handed to the Phase 5 exit review.

## 9. Reproduction

Reproduction never passes a worker-count override: `n_jobs` is part of every cell's parameter signature and
the config's contract forbids it (the runner exits 2). Parallelize across shards.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN=results/phase5-2026-09-10

# SLURM (partition/resource flags are cluster-specific); STUDY_N_JOBS must stay unset.
sbatch -p 512x1024 --cpus-per-task=1 --mem=2G --time=6:00:00 --array=0-99 \
    --export=ALL,STUDY_CONFIG=$(pwd)/examples/trajectory_power_study/phase5_power_study.json,STUDY_OUT=$(pwd)/$RUN,N_SHARDS=100 \
    scripts/motco_study_array.sbatch

# Locally, in K resumable shards (one process each):
for i in $(seq 0 $((K-1))); do
  uv run python scripts/run_study_shard.py \
      --config examples/trajectory_power_study/phase5_power_study.json \
      --out-dir $RUN --shard-index "$i" --n-shards K --error-policy record &
done; wait

uv run python scripts/motco_study.py merge  --out-dir $RUN
uv run python scripts/motco_study.py report \
    --config examples/trajectory_power_study/phase5_power_study.json --out-dir $RUN
```

At code revision `b8f7daa` with config sha256 `4f10aa1f…09826`, this reproduces the record set whose
`merged.jsonl` has sha256 `c96be03e…cb05a`. The report writes `report/report_contract.json` and
`report/driver_report.csv` beside the usual outputs and the gate's `phase4_*` files; `report/` and
`PROVENANCE.txt` are committed, and the shard and merged JSONL stay gitignored as regenerable.
