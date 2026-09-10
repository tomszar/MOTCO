## Context

See [proposal.md](proposal.md) — Why. The state of play after the paper-grade run:

- **The mechanism is already measurable from committed data.** `realized_geometry` persists per-omic scopes
  at `population_native`, `population_standardized` and `observed_standardized`, and the joint scope at all
  four checkpoints including `pls_latent`. Reading `results/phase5-2026-09-10/merged.jsonl` through
  `summarize_realized_geometry` gives, at `population_standardized` for the magnitude mode:

  | statistic | scope | e = 0.25 | 0.50 | 0.75 | 1.00 | anchor |
  |---|---|---|---|---|---|---|
  | `angle` | methylation / expression / proteomics | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0 |
  | `angle` | **joint** | 6.6978 | 12.1199 | 16.5890 | 20.3293 | 0.0 |
  | `shape` | methylation / expression / proteomics | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0 |
  | `shape` | **joint** | 0.0076 | 0.0138 | 0.0187 | 0.0226 | 0.0 |
  | `delta` | methylation | 9.9148 | 18.1365 | 24.8473 | 30.3300 | 0.0 |
  | `delta` | expression / proteomics | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0 |

  The surgery is exactly size-pure within each block; the off-target response exists only in the
  concatenation. The zeros are exact, not small — this is analytic geometry, not a noisy effect.

- **The localization instrument's asymmetry is exactly why the gate's blocking pair is unclassifiable.**
  `_normalized_geometry` divides `delta` by mean path length and `angle` by 180°, but passes `shape`
  through raw. At e = 1.00 the joint values normalize to `angle` 20.3293/180 = 0.1129 (≥ 0.05 → material,
  `construction_present`) and `shape` 0.0226 (< 0.05 → `not_material`), while the corresponding RRPP test
  rejects `shape` at 0.742. One absolute threshold cannot serve a statistic whose entire response range is
  below it.

- **`characterize_two_stage` already exists** in `specificity.py` for shape-free `n_stages=2` isolation, as
  does `evaluate_shape_null`. No new probe framework is needed.

- **The Phase 5 `report/` is committed and asserted to regenerate byte-identically** across all 23 files
  (findings report §9). Any change to the materiality rule breaks that assertion unless handled.

## Goals / Non-Goals

**Goals:**

- Record the block-decomposition finding with its numbers, in a citable form, so the exit review reasons
  from evidence rather than from a source-code reading.
- Answer the one open empirical question: does scaling every omic's δ together drive the joint `angle` and
  `shape` to the anchor's exact zero *after* per-block standardization?
- Make localization materiality commensurable across the three statistics, so `shape` responses are
  classifiable, without changing what any already-reported run says.

**Non-Goals:**

- Adopting a corrected `magnitude_kind` as a production mode, or re-running any study.
- Changing the `delta`/`angle`/`shape` estimators, RRPP, the generator's sampling model, the cross-omic
  coupling, the gate roles, the acceptance targets, or `phase5_power_study.json`.
- Making the exit review's method-versus-claim decision.

## Decisions

**D1 — The block decomposition is a reader over merged records, not a new simulation.**
The probe takes a merged JSONL path and emits a table of pooled-versus-per-block `delta`/`angle`/`shape` at
each checkpoint for a chosen mode, beside the shared zero-effect anchor. It adds no generator call.
*Alternative considered:* a dedicated probe profile that regenerates data at the design point. Rejected —
it would produce the same numbers at real cost, and reading the actual paper-grade records is strictly
better evidence for a report about the paper-grade run.

**D2 — Materiality becomes an excess in units of the checkpoint's own zero-effect null dispersion.**
The zero-effect anchor already supplies the null reference per (checkpoint, statistic); the change is to
divide the excess by that reference's dispersion across replicates (its SD over the anchor's replicates at
that checkpoint and scope) instead of comparing a raw normalized excess to 0.05. Every statistic is then
judged in the same units, and the emitted row records both the excess and the dispersion it was scaled by,
per the spec. The per-statistic normalizations (`delta` by path length, `angle` by 180°, `shape` raw) stay
as the *within-checkpoint* scale-free step; the recalibration sits on top of them.
*Alternative considered:* per-statistic absolute thresholds (e.g. 0.05 for `angle`, 0.005 for `shape`).
Rejected — three hand-tuned constants would need re-tuning at every design point, which is the same defect
in a new form. *Alternative considered:* scaling by the anchor's IQR or null q95 rather than SD. SD is used
and is recorded in the output row.

**D2a — A degenerate null dispersion means "any real excess is material", not "unclassifiable".**
Revised during implementation. The anchor's dispersion at the *population* checkpoints is analytically
zero and arrives as floating-point dust or exact zero — measured on the Phase 5 records at
`population_standardized`, joint scope: `delta` sd **exactly 0.0**, `angle` sd 8.7e-07, `shape` sd 3.5e-16.
Dividing by those is either a zero-division or a meaningless ratio (a real response would score ~1e7
"dispersion units"). Emitting an unclassifiable row instead — as this design first specified — would make
every population checkpoint unclassifiable, discarding exactly the evidence the change exists to surface.
So when the anchor's dispersion is at or below a dust tolerance, materiality falls back to "excess exceeds
the dust tolerance", which is the correct limit of the dispersion rule as the null variance goes to zero:
a null with no variance is exceeded by any real difference. The row records that the fallback applied and
why. The dispersion-units path then governs the `observed_standardized` and `pls_latent` checkpoints, where
the anchor's dispersion is substantial and real (`angle` sd 25.4 against a mean of 13.0).

**D3 — The frozen Phase 5 `report/` is protected by keeping the old rule reachable and recording which rule
produced a classification.** `localize_off_diagonal` gains an explicit rule selector; the legacy absolute
rule remains available so the committed run reproduces byte-identically, and the emitted rows name the rule.
The Phase 5 `report/` is **not** regenerated under the new rule as part of this change; the recalibrated
classification for the two failing pairs is reported in this change's own findings report, where it is
labelled as a re-analysis rather than an amendment to a predeclared result.
*Alternative considered:* switch the default and re-issue the Phase 5 `report/`. Rejected — the findings
report's byte-identical claim and the predeclared-and-frozen discipline are worth more than a tidy default,
and the exit review can adopt the new rule as the default when it decides the revision.

**D3a — The legacy rule keeps the legacy *schema*, not just the legacy verdict.**
Found during implementation: pinning the rule is not sufficient, because the new rule's reasoning needs
extra columns (`materiality_rule`, `materiality_basis`, `null_dispersion`,
`excess_in_dispersion_units`) and emitting them unconditionally rewrites every already-committed
`phase4_localization.csv`, breaking the byte-identical guarantee just as surely as changing a classification
would. So the extra columns travel with the new rule: under `"absolute"` the frame carries exactly the ten
legacy columns. Verified — the Phase 5 `report/` regenerates byte-identically across all 23 files after the
change.

**D4 — The uniform-δ candidate is a probe-only construction, gated away from study configuration.**
It is reachable from the diagnostic entry point and not from `SemiSyntheticTrajectoryParams.magnitude_kind`
as a selectable value, so no committed profile can silently acquire it and `_MAGNITUDE_KINDS` validation
keeps rejecting it. A test asserts a study config naming it is refused.
*Alternative considered:* add it as a third `magnitude_kind` now. Rejected — that is the method revision, and
adopting it before the exit review decides would pre-empt the decision this change exists to inform.

**D5 — The candidate is evaluated at the standardized-geometry checkpoint only.**
The question is whether joint `angle`/`shape` go to zero after per-block standardization; that is answered by
the analytic/standardized population geometry, with no sampling, no RRPP and no PLS fit. The comparator runs
at the Phase 5 design point's generator parameters at the same effect grid.
*Alternative considered:* a full evaluation with rejection rates. Rejected as premature — if the geometry does
not go to zero, rejection rates are moot; if it does, the exit review will want a properly designed
confirmation run, not a probe-scale one.

**D6 — The two-stage isolation is a confirmation, reported as such.** `characterize_two_stage` runs magnitude
at `n_stages=2` and the report states whether the joint orientation response persists where shape is
undefinable. Because the per-block zeros are exact, this cannot overturn D1's finding; it can only reveal that
the four-stage joint `angle` was partly an artefact of shape removal, which would be worth knowing.

## Risks / Trade-offs

- **The recalibration could flip classifications in the other direction**, making previously `construction_present`
  responses immaterial or vice versa → D3 keeps the legacy rule reachable and the reported rows name their rule,
  so any flip is visible and attributable rather than silent. The findings report must state every
  classification that changes for the Phase 5 pairs.
- **The anchor's null dispersion is degenerate at the population checkpoints** (measured: `delta` sd exactly
  0.0, `angle` sd 8.7e-07, `shape` sd 3.5e-16) → D2a: fall back to a dust-tolerance test and record that it
  applied, rather than dividing by dust or discarding the row. This is the same trap the existing "compare an
  excess, not a ratio" comment warns about, one level up, and the first version of this design fell into it.
- **The uniform-δ candidate might drive joint geometry to zero in population but not in observed data**
  → the probe reports the standardized *population* checkpoint, which is the right level for a constructibility
  question; the report must not over-read it as a power claim, and D5 says so.
- **Scope pressure toward adopting the fix** once the candidate looks clean → D4 makes it unreachable from
  configuration and a test enforces that, so adoption requires a deliberate later change.
- **The finding weakens a paper claim rather than repairing it.** If the magnitude control is unfixable
  without redefining the mode, this change will have made that plainer, not solved it. That is the intended
  outcome: the proposal's premise is that the exit review needs facts, including unwelcome ones.
