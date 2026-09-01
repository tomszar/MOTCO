# Orientation sign anchor: a real defect that does not explain the orientation shortfall

**Run date:** 2026-08-28
**Configuration:** `examples/trajectory_power_study/orientation_signfix_rerun_100x199.json`
**Records:** `results/orientation-signfix-2026-08-28/merged.jsonl` (1,500 replicates, 0 failures)
**Baseline for comparison:** `results/phase4-2026-08-27/merged.jsonl`
**Change:** `openspec/changes/fix-orientation-sign-anchor`
**Environment:** Python 3.11.15, numpy 2.3.5, pandas 2.3.3, scikit-learn 1.8.0, scipy 1.16.3; no R at runtime

## Summary

`_estimate_orientation` carried a genuine defect: PC1's sign was anchored on the **centered** first-stage
row, a quantity that vanishes for trajectories that depart and return laterally, leaving the sign to be
decided by noise. The defect is real, reproducible, and now fixed by anchoring on net displacement, with an
invariance contract in `tests/test_trajectory_orientation.py` that the previous estimator fails.

**The fix does not do what the change proposed it would.** Re-running the Phase 4 orientation, shape, and
translation cells on byte-identical data shows:

- Orientation `angle` power moves from 0.59/0.59/0.64/0.65 to **0.64/0.64/0.67/0.68** — a uniform ~3-point
  lift that leaves the curve flat and well under the 0.80 floor.
- The 150–180° mass in null cells is **unchanged**: 21 of 700 before, 21 of 700 after.
- The realized-geometry checkpoints substantially stand: population orientation 81.1° → 80.5°, PLS latent
  57.5° → 51.1°.

The proposal's central claim — that reported orientation power is an artifact-suppressed underestimate
explaining "the shortfall's every symptom" — is **contradicted by its own re-run**. Phase 5 readiness item 1
stays open and `diagnose-angle-null-pivotality` is not displaced.

## The defect and its mechanism

The estimator is a port of Adams & Collyer's phenotypic trajectory analysis
(`tests/data/reference/evo_649_sm_suppmat.r`). Orientation is PC1 of the centered stage configuration
(`:57`); PC1's inherent sign ambiguity is resolved at `:64` by a line commented `#check startingpoint
location`, which anchors on the raw first-stage row.

MOTCO centered the configuration first and then anchored on the **centered** row, computing
`PC1 · (stage₁ − centroid)`. That quantity vanishes whenever a trajectory departs and returns laterally
along its own principal axis — precisely what a bend produces.

On the constructed four-stage geometry in `tests/test_trajectory_orientation.py` (`BENT_TRAJECTORY`), whose
first stage projects **exactly zero** onto its own centered principal axis:

- the sign flips in **98 of 200** perturbation draws at noise sd 0.01, against a trajectory extent of ~5;
- two copies carrying no orientation difference are measured at a **median 179.4°**.

Both are pinned as tests (`test_orientation_sign_is_stable_under_small_perturbation`,
`test_bent_identical_trajectories_report_zero_angle`), and both fail against the previous estimator.

## The anchor decision

The sign is now anchored on net displacement, `PC1 · (stage_last − stage_first)`. PC1 itself is unchanged.

**Why not the faithful port.** Anchoring on the raw first-stage row makes the sign depend on where the
trajectory sits relative to the coordinate origin, so a pure translation can reverse it. Four regimes,
pinned in `test_four_regime_anchor_comparison`:

| regime | truth | centered (previous) | raw (faithful) | net displacement |
|---|---:|---:|---:|---:|
| bent 4-stage, identical groups | 0° | **180°** | 0° | 0° |
| straight 2-stage, groups either side of origin | 0° | 0° | **180°** | 0° |
| translation control, group B shifted | 0° | 0° | **180°** | 0° |
| genuine 90° difference | 90° | 90° | 90° | 90° |

The faithful port trades one failure for two, and one of the two is MOTCO's own translation null control —
which Phase 4 passes cleanly at every effect size. Adopting it would break a working control to fix a
broken one.

**Accepted residual degeneracy.** Net displacement vanishes for a closed trajectory whose last stage returns
to its first. The evidence below suggests this is not merely theoretical in the study's latent space.

## Fixture equality — correcting the change's own claim

`design.md` originally asserted that all three anchors reproduce both committed R fixtures exactly. Measured,
that is false for `example1`. Maximum `|angle − R|` over every reported pair:

| fixture | centered (previous) | raw (faithful) | net displacement |
|---|---:|---:|---:|
| `results_example2.csv` (5 levels) | 9.9e-12 | 9.9e-12 | 9.9e-12 |
| `results_example1.csv` (2 levels) | **30.60** | 4.8e-13 | **30.60** |

MOTCO has never reproduced `example1`'s `t1/t3` and `t2/t3` angles: it reports 74.70°/76.49° against R's
105.30°/103.51°, because those pairs straddle the PCA origin and R anchors on the raw row.
`tests/test_permutation.py` has always accepted `180 − exp_angle`, which is why this went unrecorded for the
project's entire history.

This change neither introduces nor removes that deviation — net displacement reproduces the shipped values
exactly. The deviation is itself an instance of the origin-side dependence that rules the raw anchor out.
The claim has been corrected in `design.md` and `proposal.md`.

## Corrected operating characteristics

Re-run scope follows `design.md`: orientation, shape, and translation cells, plus the shared zero-effect
anchor and both Type I baselines. Generator, evaluation, replicate count, effect sizes, base seed, and
matched-seed policy are identical to the Phase 4 pilot; **all 1,500 replicate generator seeds were verified
byte-identical to Phase 4's**, so every comparison below is paired on the same data.

### Orientation `angle` power against the 0.80 floor

| effect | Phase 4 | corrected | MC SE |
|---:|---:|---:|---:|
| 0.25 | 0.590 | **0.640** | ±0.048 |
| 0.50 | 0.590 | **0.640** | ±0.048 |
| 0.75 | 0.640 | **0.670** | ±0.047 |
| 1.00 | 0.650 | **0.680** | ±0.047 |

Still flat, still short of 0.80 by more than two standard errors. **The floor is not met.** The study's own
gate machinery reaches the same verdict independently: `report/phase4_gate_decision.json` records
`decision: hold`, `mandatory_power[orientation,angle] (top_rate=0.680 < floor=0.800)`.

### The translation control still holds

| cell | Phase 4 | corrected |
|---|---:|---:|
| `translation` 0.25 | 0.030 | 0.030 |
| `translation` 0.50 | 0.070 | 0.060 |
| `translation` 0.75 | 0.050 | 0.030 |
| `translation` 1.00 | 0.050 | 0.030 |
| `none` (power anchor) | 0.040 | 0.040 |
| `none` (Type I baseline) | 0.040 | 0.050 |
| `translation` (Type I baseline) | 0.040 | 0.040 |

Nominal level everywhere. The raw-anchor trap was correctly avoided.

### Shape-cell `angle` is essentially unchanged

0.510 → 0.580, 0.690 → 0.710, 0.770 → 0.790, 0.830 → 0.820 across effects 0.25 → 1.00.

### The null-cell artifact is not eliminated

Pooled `none` + `translation` cells (n = 700, true angle exactly 0°):

| run | <30° | 30–60° | 60–90° | 90–135° | 135–150° | **150–180°** | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Phase 4 | 663 | 8 | 1 | 3 | 4 | **21** | 177.8° |
| corrected | 663 | 7 | 2 | 1 | 6 | **21** | **178.7°** |

Identical in aggregate, with a slightly higher maximum.

**The paired replicate-level view shows why.** Of the 700 null replicates, 654 are unchanged and **46 flip to
their supplement** (θ → 180 − θ). Of those flips, **21 cross into >150° and 21 cross out** — an exact wash.
The corrected anchor does not stop the sign from being noise-decided in the study's latent space; it
reshuffles which replicates it happens to. This is the signature of `PC1 · (stage_last − stage_first)` itself
landing near zero — the residual degeneracy accepted above — and it means the study-path artifact has a
driver this change does not address.

### Realized-geometry checkpoints (task 4.5)

| checkpoint | Phase 4 | corrected |
|---|---:|---:|
| orientation @ 1.00, population standardized | 81.1° | 80.5° |
| orientation @ 1.00, observed standardized | 81.0° | 80.8° |
| orientation @ 1.00, PLS latent | 57.5° | 51.1° |

The previously reported 81°/57° figures are **not** overturned. Arguments resting on "the construction
survives integration" stand.

## Conclusions

1. The sign-anchor defect was real and is fixed, with an invariance contract the previous estimator fails.
   This stands on its own as a bug fix.
2. It is **not** the cause of the Phase 4 orientation shortfall. Power rises ~4 points and stays flat at
   0.69 against a 0.80 floor.
3. It is **not** the source of the study-path near-180° null angles, which are unchanged at 21/700.
4. Phase 5 readiness item 1 remains **blocking and open**.
5. `diagnose-angle-null-pivotality` is **not** displaced and should proceed on its own terms.
6. A new open question, raised by this run and not answered by it: what decides the orientation sign in the
   latent space when net displacement is itself small, and is PC1 direction (not merely its sign) unstable
   when the top two singular values of the stage configuration are close?

## What is out of scope here

Per `design.md`, the `magnitude` cells were not re-run: magnitude's `delta` conclusion does not depend on the
orientation estimator. Its `angle` and `shape` control rows therefore carry the previous estimator's values
and were not re-measured. The full Phase 4 gate was not re-evaluated for the same reason.

## Provenance

- Code revision: this change's branch, parent `4725840`.
- Estimator: `src/motco/stats/trajectory.py::_estimate_orientation`.
- Invariance contract: `tests/test_trajectory_orientation.py` (23 tests).
- Comparison tooling: `scripts/orientation_signfix_analysis.py`; its full output is committed as
  `results/orientation-signfix-2026-08-28/report/comparison_vs_phase4.txt`.
- Study report outputs (committed): `results/orientation-signfix-2026-08-28/report/`, including
  `acceptance_report.csv` and `phase4_gate_decision.json`.
- Rejection rates use the study's strict `p < alpha` convention (`grid._is_rejection`).
- Every figure above traces to those tests or to the two merged record sets named in the header.
