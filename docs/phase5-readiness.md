# Phase 5 readiness worklist

Work completed before launching the paper-grade study (which ran 2026-09-10). Derived from the
[Phase 4 medium PLS pilot](reports/phase4-medium-pls-pilot-2026-08-27.md) (run 2026-08-27, gate decision
**HOLD**). Every number below comes from `results/phase4-2026-08-27/report/`.

**Status: item 1 resolved 2026-09-01, item 2 resolved 2026-09-03, item 3 resolved 2026-09-08
([latent-rank ladder report](reports/latent-rank-ladder-2026-09-08.md); committed rank rule: stage-supervised
double CV), item 4 resolved 2026-09-08, item 5 resolved 2026-09-09 (report contract, paper-grade profile
`examples/trajectory_power_study/phase5_power_study.json`, and report template committed). All five items are
closed, and the paper-grade run executed 2026-09-10 — gate decision **HOLD** on the two magnitude
mandatory controls, with all three mandatory power diagonals and all Type I checks met
([findings report](reports/phase5-paper-grade-2026-09-10.md)), whose causes were then established by the
[magnitude-construction diagnostic](reports/magnitude-construction-diagnostic-2026-09-10.md). This worklist
is closed, and the [Phase 5 exit review](reports/phase5-exit-review-2026-09-10.md) closed the exit gate on 2026-09-10 as a conditional pass.** Phase 4 is complete; these are its follow-ups. The blocking item is closed — `angle` proceeds as
specified — and the 0.80 orientation power floor, handed to item 4, is **met at the chosen design point
ρ = 0, n = 1200** (orientation `angle` power 0.88, 1·SE lower bound 0.85; see the
[design-point pilot report](reports/phase5-design-point-pilot-2026-09-08.md)). See also
[the pivotality report](reports/angle-null-pivotality-2026-09-01.md). The
[geometry audit](reports/geometry-audit-2026-09-01.md) (2026-09-01) narrowed items 2–4 and added
preconditions — see its remediation plan (P1–P6); P3 landed 2026-09-03 and closed item 2, and P1/P2/P4 were
the preconditions item 4's pilot ran on.

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

**Closed 2026-09-02.** It does now: the relative eigengap of the centered latent stage-mean configuration is
persisted on every replicate (`config_spectrum`, pooled and per group, plus the pooled eigengap over the
permutation draws), and the pivotality analysis reports its association with each replicate's own null
width. See [Recording the latent configuration
spectrum](reports/latent-config-spectrum-2026-09-02.md).

## 2. Investigate orientation → shape at the PLS checkpoint — **resolved 2026-09-03**

> **Resolved.** The reflection-policy precondition is closed and the rank-limited-projection account is
> supported by direct measurement. **Decision: Phase 5 predeclares orientation's `shape` response as a
> projection artifact of the rank-3 stage-supervised latent space.** See the two bolded findings at the end
> of this item.

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
vacuous), so the localization table's rows mixed two contracts.

**Reflection-policy precondition: closed on 2026-09-03** by `unify-shape-reflection-policy` (plan item P3).
The policy is now uniform — the Procrustes alignment optimizes over the full orthogonal group, so reflections
are aligned away at *every* ambient dimension and `shape` means the same thing at every checkpoint by
construction. The localization table no longer mixes contracts, and Phase 5 may make cross-checkpoint shape
claims without a per-row regime annotation. `SHAPE_STATISTIC_VERSION = 2` enters the study parameter
signature, so pre-change shards refuse to resume under the new estimator; historical committed outputs keep
their recorded values and comparisons against them must note the contract change. Measured cost, as the audit
predicted: none on real geometry.

**Rank-limited-projection account: supported** by the follow-up probe
([`results/latent-rank-probe-2026-09-03/`](../results/latent-rank-probe-2026-09-03/latent_rank_probe.md),
driver `scripts/latent_rank_probe.py`). Holding the Phase-4 pilot design point fixed with matched seeds
(n = 300, 4 stages, effect 1.00, 100 replicates, 199 permutations) and varying **only** the retained PLS rank
via the new diagnostic `forced_components` override, the orientation→shape response decays steeply as rank
grows — monotone through rank 9, then flat — and its rejection rate collapses with it:

| retained rank | 3 (CV-selected) | 4 | 6 | 9 | 12 |
|---|---|---|---|---|---|
| observed `shape` | 0.0694 | 0.0541 | 0.0195 | 0.0135 | 0.0136 |
| rejection rate | 0.97 | 1.00 | 0.36 | 0.08 | 0.09 |

The response falls by 80% across the ladder and passes through the standardized-observed population value
(0.0424) between ranks 4 and 6, while the `none` control is flat from rank 4 onward (0.0068 → 0.0065,
rejection 0.00) — so the decay is specific to the orientation surgery, not generic to rank. The latent and
population values are not on a common scale, so the crossing rank is indicative; the load-bearing evidence is
the decay and the rejection-rate collapse.

**Item 2 is therefore resolved for the purpose it was raised:** Phase 5 predeclares orientation's `shape`
response as a **projection artifact of the rank-3 stage-supervised latent space**, not a finding about the
constructions. Note this makes item 3 (latent dimensionality) sharper rather than moot — the same evidence
shows what a rank-3 space costs the group contrast.

## 3. Reconsider latent dimensionality for group contrasts — **resolved 2026-09-08**

> **Resolved.** The latent-rank ladder ([report](reports/latent-rank-ladder-2026-09-08.md), run 2026-09-08,
> `results/phase5-latent-rank-2026-09-08/`, 8,000 units, 0 failures, 0 censored surgeries) held the chosen
> design point fixed (ρ = 0, n = 1200, four stages, `p_dmp = 0.1`) and varied only the retained PLS rank over
> {CV, 3, 4, 6, 9, 12} for all four modes, every column measuring the *same* generated datasets (evaluation-only
> design axis, shared matched-seed family). The predeclared rule (`acceptance.rank_decision`: orientation
> `angle` gain > 2 pooled SE, anchors within α + 2 SE, magnitude `delta` and shape `shape` loss ≤ 2 pooled SE)
> returned **`keep_cv`** (`report/rank_decision.json`).
>
> **Committed Phase 5 rank rule (group-blind): stage-supervised double cross-validation** (`plsda_doubleCV`,
> modal LV across repeats, parsimony tie-break) — the production rule, unchanged. At the design point it
> selects rank 3 = `n_stages − 1` (range 2–3 over 1,500 CV units). No group label enters the sizing, so the
> fixed-latent-space RRPP conditioning the geometry audit protects is intact.
>
> **What the ladder showed.** Orientation `angle` power at e = 1.00 is 0.85 (CV), 0.86 (rank 3), and
> 0.79–0.81 at ranks 4–12: no rank buys `angle` power. Every rank above 3 loses power elsewhere — shape
> `shape` 1.00 → 0.57 and orientation `shape` 0.99 → 0.60 by rank 9, orientation/shape `delta` 0.9 → 0.5 —
> while the off-target `angle` response *rises* (magnitude → `angle` 0.24 → 0.81, shape → `angle` 0.79 →
> 0.95): extra components carry group-specific noise directions that rotate PC1. The anchor's `angle` null
> q95 nearly doubles (5.9° → 10.8°) at rank ≥ 4 while the recorded eigengap stays ≈ 0.047, i.e. the
> measurement space acquired noise, not geometry. No anchor is inflated (≤ 0.03 at every rank; `delta`/`shape`
> become conservative, 0/100, at rank ≥ 4). Forced 3 equals CV within 0.03 everywhere, so CV selection noise
> is immaterial.
>
> **Re-measurement at the chosen point under CV (`p_dmp = 0.1`):** magnitude `delta` 1.00 at every effect,
> shape `shape` 0.95 / 0.97 / 1.00, orientation `angle` 0.89 / 0.89 / 0.85 (agrees with the design-point
> pilot's 0.88 within one pooled SE), orientation → `shape` 0.99–1.00 (does not decay at CV rank; decays only
> as rank grows, as the probe found). The item-2 predeclaration for orientation's `shape` response stands.
>
> The ~6% retained orientation contrast recorded below is therefore not a power deficit to fix by
> re-sizing: the `angle` test reads what it needs from the `n_stages − 1` space, and the remaining
> variance is stage-unorganized noise.

The history of how this item was scoped is retained below.

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

**P1 landed 2026-09-02.** The measure is now recorded per replicate rather than regenerated: every record
carries `config_spectrum` (pooled and per-group normalized spectra and relative eigengaps, plus the pooled
eigengap over its own permutation draws), the study report stratifies orientation power by it, and the
pivotality analysis reports its association with the null width. Re-sizing the latent space can now be
argued from recorded evidence — see
[Recording the latent configuration spectrum](reports/latent-config-spectrum-2026-09-02.md). The
group-aware-supervision caution stands: this change only *observes* the spectrum.

## 4. Choose the Phase 5 design point — **resolved 2026-09-08**

> **Resolved.** The design-point pilot ([report](reports/phase5-design-point-pilot-2026-09-08.md), run
> 2026-09-08, `results/phase5-design-point-2026-09-08/`, 6,500 units, 0 failures, 0 censored surgeries)
> crossed baseline continuity ρ ∈ {0, 0.5, 0.8} with `n_samples` ∈ {300, 600, 1200} at `p_dmp = 0.1`, four
> stages, 100 replicates × 199 permutations, on an effect axis (0.25–1.00) that is uncensored at every
> design point.
>
> **Decision: the Phase 5 design point is ρ = 0 (independent baseline), n = 1200 (300 samples per
> group-stage cell), four stages, `p_dmp = 0.1`, pooled PLS on M-value methylation with CV-selected
> rank.** Orientation `angle` power at e = 1.00 is **0.88** (MC SE 0.032, 1·SE lower bound 0.85) and the
> column's own zero-effect anchor is at nominal level on all three statistics (0.04 / 0.04 / 0.03). The 0.80
> floor is **met, not revised**, at the isotropic stress-test endpoint — the most general claim the grid
> offered (`report/design_point_decision.json`).
>
> **What the covariates showed.** Along n at fixed ρ = 0 the eigengap is constant (≈ 0.05) while the median
> `angle` null width contracts 40.7° → 18.8° → 10.8° and power rises 0.55 → 0.76 → 0.88: sample size shrinks
> the null-width dispersion exactly as item 1 predicted. **Continuity does not help and is non-monotone:**
> ρ = 0.5 is worse than ρ = 0 at every n (0.24 / 0.46 / 0.69) and ρ = 0.8 beats ρ = 0 only at n = 300. The
> eigengap rises with ρ as designed, and at ρ = 0.5 the null is the narrowest in the grid, but the same
> nominal orientation surgery realizes a *smaller* latent contrast on a trending baseline (median observed
> angle 28° at ρ = 0.5, n = 1200, against 66° at ρ = 0). The Phase 5 claim is therefore n-conditional; the
> eigengap distribution at the chosen point (median 0.049, terciles 0.038–0.063) is the observable to report
> beside any real-data orientation result, and baseline continuity must not be used as a lever to buy power.
>
> **Hand-off to item 3.** Selected rank is 3 (`n_stages − 1`) at the chosen point for every n (range 2–4),
> so the retained-rank question is unchanged by sample size. New fact for item 3: only a strongly trending
> baseline (ρ = 0.8) moves the CV off `n_stages − 1` — orientation at e = 1.00 there selects median rank 7–8
> (up to 14) against the column's anchor at 4, and it is the only ρ at which orientation's `delta`/`shape`
> responses fall, consistent with the latent-rank probe's decay.
>
> **Phase 5 must re-measure magnitude and shape** at the chosen point: `p_dmp` changed from 0.2 to 0.1, and
> the pilot deliberately ran only orientation and translation.

The history of how this item was scoped is retained below. The Phase 4 pilot used n = 300 with four stages
(75 samples per group-stage cell) over ~660 standardized features.

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

**The censored effect axis is fixed (P2, 2026-09-03).** The
[geometry audit](reports/geometry-audit-2026-09-01.md) finding F2 — the relocation clamp saturating the
orientation surgery at e ≈ 0.69, leaving 80 of 100 replicate pairs at e = 0.75 and e = 1.00 byte-identical —
was the precondition on this item. The generator now applies an explicit censoring policy
(`generator.surgery_censoring`, default `"error"`): a surgery that cannot be realized in full fails loudly
instead of clamping, truth metadata records nominal-vs-realized size and a `censored` flag, study
enumeration rejects over-headroom cells before any compute is spent, and `report/realized_surgery.csv`
flags duplicated constructions from the records alone.

**What this item must now decide:** which effect axis replaces the censored one. Enumeration reports the
saturating effect per cell — at the pilot's `p_dmp = 0.2` with four stages that is **e ≈ 0.56 for
orientation and e ≈ 0.29 for translation** (3σ guard band included). A Phase 5 axis must sit under those
bounds, or move them: lower `p_dmp`, use fewer stages, or change the surgery so its destination pool is not
the complement of the stage program. Every config in `examples/trajectory_power_study/` predates the policy
and carries `"surgery_censoring": "clamp"` as a historical record — **new configs must not copy it.**

**The second lever now exists (P4).** Besides sample size, the audit handed this item baseline
continuity: n-scaling shrinks the noise term, but the eigengap's lower tail (near-isotropic baseline draws)
is what caps the curve, and that tail is a property of the baseline construction, not of n. The generator
now exposes it as a declared axis — `generator.baseline_continuity` (ρ ∈ [0, 1)) makes each CpG's per-stage
differential status follow a stationary first-order Markov chain, so the per-stage Bernoulli(`p_dmp`)
marginal (and therefore per-stage counts, δ semantics, and the cross-omic coupling) is unchanged along the
axis while stage means acquire a *trend*: pairwise distances grow with stage separation, giving the
configuration a dominant PC1. ρ = 0 is the current independent baseline, byte-identical, and remains the
declared isotropic stress-test endpoint.

Two consequences for the design-point study. First, the surgery headroom moves with ρ: the expected
stage-active union is `1 − (1 − p_dmp)·(1 − p_dmp·(1 − ρ))^(n_stages − 1)`, so overlapping stage programs
*enlarge* every pool-limited surgery's destination pool. At `p_dmp = 0.2` with four stages the saturating
effect moves from **e ≈ 0.56 (orientation) / 0.29 (translation) at ρ = 0** to **e ≈ 2.77 / 1.47 at ρ = 0.9**
(3σ guard band included), so the usable effect axis this item must choose widens along ρ. Enumeration
already reads each cell's own ρ, so an effect rejected at ρ = 0 may enumerate at higher ρ. Second, when a study sweeps the axis,
`report/continuity_resolved_orientation.csv` presents orientation rejection rates, the recorded eigengap
distribution, and the `null_summary["angle"]["q95"]` dispersion per continuity value — so power differences
along ρ are read off the recorded geometry, which is the observable that carries the claim to real data,
rather than off the knob. **Which ρ values Phase 5 sweeps is this item's decision.**

**The stratifying covariate is available (P1, 2026-09-02).** The design-point study can measure how the
`null_summary["angle"]["q95"]` dispersion contracts with samples per group-stage cell *stratified by the
geometry that governs it*: every record now carries `config_spectrum`, and
`report/eigengap_stratified_power.csv` reports orientation power within eigengap terciles per cell. See
[Recording the latent configuration spectrum](reports/latent-config-spectrum-2026-09-02.md).

## 5. Carry into the Phase 5 report contract — **resolved 2026-09-09**

> **Resolved** by `phase5-report-contract-and-config`. The four items below are no longer prose: they are a
> declared `report_contract` block in the committed paper-grade profile
> [`examples/trajectory_power_study/phase5_power_study.json`](../examples/trajectory_power_study/phase5_power_study.json)
> (`driver_component: observed`, `cross_replicate_driver_agreement: descriptive`, `n_jobs_override: forbid`),
> each **enforced or echoed** by the study code, and the dated Phase 5 findings report must follow the committed
> [report template](../examples/trajectory_power_study/phase5_report_template.md), which carries a section
> for every item. The loader also rejects unknown top-level config keys, so a misspelled contract fails loudly.
> How each point is enforced:
>
> | item | contract field | enforced / echoed by |
> |---|---|---|
> | observed-component drivers | `driver_component: observed` | `report/driver_report.csv` (declared component only) and the attribution figure ("Within-replicate bootstrap stability (observed component)"); `phase4_attribution.csv` keeps all three components as evidence |
> | no cross-replicate stability claim | `cross_replicate_driver_agreement: descriptive` (the only legal value; `claim` is rejected) | `top_k_jaccard` / `sign_agreement` stay in `phase4_attribution.csv`, are absent from the driver table and figure, and `report/report_contract.json` states they are not a stability claim |
> | `n_jobs` in the signature | `n_jobs_override: forbid` | `scripts/run_study_shard.py` exits 2 before enumeration when `--n-jobs` (`STUDY_N_JOBS`) differs from `evaluation.n_jobs`; the echo records the uniform `n_jobs` the records carry and refuses a mixed set |
> | one shared anchor | (matched seeds, echoed) | `report/report_contract.json` names the anchor `cell_id`, the modes it resolves, and `counted_as: 1`; operating rows stay flagged `from_shared_anchor` |
>
> The profile itself is the ladder's cross-validated column at paper-grade precision (500 × 999, five-point
> effect axis, gate enabled with the Phase 4 roles, acceptance specificity = the gate's mandatory controls —
> orientation → `shape` is predeclared cross-talk, not a target). The historical `study.json` is superseded.

Small items, already evidenced, that should be settled before the study rather than discovered during it.

- **Driver reports must use the observed component**, not `pls_captured` (precision 1.00 vs 0.15).
- **Do not claim cross-replicate driver stability.** Top-20 Jaccard is 0.02–0.05 and sign agreement ~0.68,
  but that is expected: each replicate index draws a fresh indicator set, so the true driver set genuinely
  differs. Matched seeds pair cells *within* a replicate index, not across them. A stability claim needs a
  design that holds the driver set fixed.
- **`n_jobs` is part of the cell parameter signature.** RRPP seeds one RNG stream per worker, so the worker
  count changes the realized permutation draws. Phase 5 must run at the config's value; parallelize across
  shards. The sbatch script forwards `--n-jobs` only when `STUDY_N_JOBS` is set — and the Phase 5 profile
  refuses it.
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
