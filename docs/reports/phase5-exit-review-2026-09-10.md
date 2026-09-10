# Phase 5 exit review — 2026-09-10

**Decision: Phase 5's exit gate is CLOSED as a conditional pass.** Every exit-gate criterion is met. The
Phase 4 gate's **HOLD** stands as recorded and is resolved as a construction defect with a known fix, not as
a defect in the `delta`/`angle`/`shape` estimators. The magnitude specificity claim is **withheld** pending
re-measurement of the corrected construction. Phase 6 is authorised to start. Phase 7 stays deferred.

**Inputs.** [Paper-grade findings report](phase5-paper-grade-2026-09-10.md) (run
`results/phase5-2026-09-10/`, 19 cells × 500 replicates × 999 permutations = 9,500 units, job 880599) and
the [magnitude-construction diagnostic](magnitude-construction-diagnostic-2026-09-10.md) (run
`results/magnitude-construction-2026-09-10/`). No new measurement was made for this review; it decides on
what those two produced.

## 1. The exit-gate criteria are met

The roadmap's Phase 5 exit gate has three criteria. Each is satisfied independently of the Phase 4 gate's
verdict, and the distinction matters: the Phase 4 gate asks whether the *statistics* met predeclared
operating targets, while the exit gate asks whether *Phase 5 produced publishable evidence*.

| criterion | status | evidence |
|---|---|---|
| Conclusions are stable at paper-grade Monte Carlo precision | **met** | Every quantity agrees with two independent matched-seed pilot families within Monte Carlo error, including both failures (magnitude/`angle` 0.276 vs the ladder's 0.24; magnitude/`shape` 0.742 vs 0.74). The 22-fold increase in Monte Carlo work changed no conclusion. |
| All deviations from preregistered targets are explained | **met** | Four deviations, each with a mechanism and a named revision — §3 below. None is a Monte Carlo sample-size question, and none was addressed by raising the replicate count (which the Phase 4 exit gate forbids). |
| The report distinguishes statistical operating characteristics from biological construction cross-talk | **met** | This is precisely what the diagnostic settled: the magnitude off-target response is construction geometry (exactly size-pure within every omic block, `angle` 0.0000 at `population_native`), not estimator behaviour. |

## 2. What Phase 5 establishes, and what it withholds

**Claimable now**, at paper-grade precision:

- **Type I error control** on all three statistics: 0.040–0.062 across both baseline cells against α = 0.05,
  every one within 2·SE (`report/acceptance_report.csv`, 6/6 targets met), and all 21 gate inflation checks
  met.
- **Translation invariance**: a constant observed-space offset moves none of the three statistics at any
  effect (0.028–0.056 across the whole sweep). This is the estimators' invariance contract, confirmed.
- **Power** on all three diagonals against the 0.80 floor: magnitude/`delta` 1.000, orientation/`angle`
  0.840 ± 0.016, shape/`shape` 0.996.
- **The measurement architecture**: features → PLS latent space → trajectory geometry, with the
  stage-supervised double-CV rank rule selecting rank 3 = `n_stages − 1` in all 19 cells.
- **Operational reproducibility**: 9,500/9,500 units present once, 0 signature mismatches, 0 failures, 0
  censored surgeries, and a committed `report/` that regenerates byte-identically from the merged records.

**Withheld pending re-measurement:**

- **The specificity claim for `angle` and `shape` against a pure size change.** The construction used to
  test it was not a pure size change (§3), so the test was invalid rather than failed. The claim is neither
  asserted nor abandoned; it awaits the corrected construction. The paper's specificity section cannot be
  written until then.

**Not claimable, on independent grounds:**

- **Cross-replicate driver stability** — no design holds the driver set fixed, so it is descriptive only.
- **A strictly-monotone orientation power claim** — see §3.

## 3. Resolution of each deviation

**D1 — `mandatory_control[magnitude,angle]` 0.276 > 0.0695 (10.3 SE). Method revision.**
`magnitude_kind='all'` scales `delta_methyl` alone while the measurement space standardizes and concatenates
all three omic blocks, so group B grows in one block and not the others and the pooled trajectory rotates by
construction. Within each omic block the surgery is exactly size-pure (`angle` 0.0000 and `shape` ~1e-17 at
`population_native`, every effect); the response appears only in the joint scope (`angle` 6.70° → 20.33°) and
is confirmed shape-free at two stages (19.05° vs 18.91° at four). A uniform-δ construction that scales every
omic's δ together is exactly size-pure in the joint space after per-block standardization (joint `angle`
8.5e-07 → 2.6e-06, joint `shape` ~1e-15 — the anchor's own floor — while joint `delta` grows 14.96 → 44.80).
**The `angle` estimator is not implicated.** Remedy: adopt the corrected construction, calibrate its effect
axis, re-measure. Tracked as its own change (§5).

**D2 — `mandatory_control[magnitude,shape]` 0.742 > 0.0695 (34.4 SE). Method revision, same fix.**
Same mechanism. The paper-grade report named a diagnostic prerequisite — whether the 0.05 normalized
materiality threshold was calibrated for the `shape` statistic's scale — and it is **discharged**: it was
not. `localize_off_diagonal` normalized `delta` by path length and `angle` by 180° but passed `shape` through
raw, so one absolute cut could never classify a statistic whose entire response range is ~1e-2. The
recalibrated null-dispersion rule classifies magnitude/`shape` as `construction_present` at every effect
while leaving translation `not_material` on all eight rows. The response was real all along and the
instrument was hiding it.

**D3 — `power[orientation,angle]` acceptance target not met (monotonicity). Claim revision.**
The 0.80 floor is cleared (0.840). The strictly-monotone form is not supported: the curve saturates from
e = 0.25 (0.842, 0.878, 0.872, 0.840) because the `angle` null widens with the observed statistic — median
q95 5.5° at the anchor to 11.3° at e = 1.00, with the IQR growing faster (4.9° → 13.9°). This is the
pivotality property already established in 2026-09-01, showing its consequence at paper grade. **Adopted
claim:** a *saturating* orientation response that clears the floor, stated with the eigengap stratification
(0.754 / 0.886 / 0.880 by tercile), never as a monotone power curve. No replicate count or rank change
addresses it — the ladder already returned `keep_cv` over five fixed ranks; a different pivot would.

**D4 — orientation/`shape` predeclared "projection-associated". Claim revision. Refuted.**
The label does not survive. Orientation's response is present *within* each omic block (max block `angle`
90.08° against joint 89.93°; max block `shape` 0.108 against joint 0.059), and under the recalibrated
materiality rule **no pair anywhere in the study localizes as projection-associated** — the paper-grade run's
single such classification, orientation/`delta` at e = 1.00, becomes `construction_present` once the
population checkpoint is judged against its own null. **Adopted claim:** orientation → `shape` is
construction-present cross-talk of a global per-omic feature permutation. No gate impact — both orientation
off-diagonals are descriptive roles. The projection-associated framing must not be repeated.

## 4. Phase 6 is authorised, with one binding condition

Phase 6 (real-data case study) may start. The HOLD concerns a synthetic construction; real cohorts have no
surgery modes, so nothing about D1/D2 blocks applying the validated configuration to real data.

**Binding condition — report orientation power against the cohort's own recorded eigengap, never the pooled
figure.** Pooled orientation/`angle` power at e = 1.00 is 0.840, but stratified by eigengap tercile it is
0.754 / 0.886 / 0.880: the lowest tercile sits *below* the predeclared 0.80 floor. The eigengap, not n and
not ρ, is the observable that transfers. A cohort landing in that tercile does not inherit 0.840.

Two stated limitations travel with any Phase 6 orientation result:

- The claim is **n-conditional**: established at n = 1200 (300 per group-stage cell) at the isotropic
  endpoint ρ = 0. Along n at fixed ρ the eigengap is constant while the `angle` null width contracts.
- The orientation surgery's realized latent contrast is **not ρ-invariant** (median observed angle 28° at
  ρ = 0.5 vs 66° at ρ = 0), so baseline continuity is not a lever that buys power.

## 5. Phase 7 stays deferred — question closed

SNF-native trajectory statistics remain deferred pending graph-native metrics. Nothing in the paper-grade run
or the diagnostic bears on SNF: the study measured the PLS latent space throughout, and `concat` appeared only
as a baseline. The deferral is therefore **not** an open question awaiting this review's evidence — it awaits
a decision to define graph-native path-length and angle statistics, which is independent work. Recording it
as closed here so it stops reading as pending.

## 6. Consequent work, in order

1. **Adopt the size-pure magnitude construction** (its own change). Promote the uniform-δ probe from
   diagnostic-only to a production `magnitude_kind`; it is currently unreachable from configuration and
   refused by the loader by design. Calibrate its effect axis first — at e = 1.00 it reaches joint `delta`
   44.80 against production's 18.14, so the two effect axes are not comparable and its power curve cannot be
   read beside the existing one without recalibration. Then re-measure the magnitude controls and lift the
   withheld specificity claim or report it as failed on its own terms.
2. **Phase 6 real-data case study**, under the §4 condition.
3. **Consider making the null-dispersion materiality rule the default.** It is opt-in here so the committed
   Phase 5 `report/` keeps regenerating byte-identically. Switching the default is a reporting change that
   should come with a recorded re-issue of any affected localization output — and note the interpretive
   consequence: because population checkpoints carry analytically zero nulls, the rule answers "is this
   construction exactly pure?" rather than "is the impurity large enough to matter". Severity is read from
   the response magnitude, not the label.

## 7. What this review does not do

It adopts no construction change, re-runs no study, and edits no predeclared artefact — the config, gate
roles, acceptance targets, and report contract stay frozen, and the committed Phase 5 `report/` is untouched.
The conditional pass is a statement about what Phase 5 established, not a waiver of the Phase 4 gate's
verdict, which stands as recorded in `results/phase5-2026-09-10/report/phase4_gate_decision.json`.
