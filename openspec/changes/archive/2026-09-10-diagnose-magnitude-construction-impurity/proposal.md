## Why

The [paper-grade Phase 5 run](../../../docs/reports/phase5-paper-grade-2026-09-10.md) returned gate decision
**HOLD**, carried entirely by the two magnitude mandatory controls: magnitude/`angle` 0.276 and
magnitude/`shape` 0.742 against a 0.0695 bound. Both reproduce values the latent-rank ladder already
measured (0.24, 0.74), so precision is not the issue and re-running is forbidden by the Phase 4 exit gate.
The Phase 5 exit review must now choose a **method revision** (make the magnitude construction size-pure)
or a **claim revision** (drop the specificity claim and report the response as construction impurity) — and
it currently cannot, because three facts are missing:

1. **The mechanism is established but unreported.** Reading the committed
   `results/phase5-2026-09-10/merged.jsonl` through `summarize_realized_geometry` settles it: at the
   `population_standardized` checkpoint the magnitude mode's realized `angle` and `shape` are **exactly
   0.0000 within every individual omic block** at every effect, while the **joint** (concatenated) scope
   carries `angle` 6.70° → 20.33° and `shape` 0.0076 → 0.0226 across e = 0.25 → 1.00; `delta` moves only in
   methylation (9.91 → 30.33) and is exactly 0.0000 in expression and proteomics. The surgery is therefore
   *perfectly size-pure per block* — `_magnitude` with `magnitude_kind='all'` (`semisynthetic.py:415`)
   scales `delta_methyl` alone — and the entire off-target response is the geometry of concatenating one
   block that grew with two that did not (`evaluation.py:579` standardizes and concatenates all three).
   It is not nonlinearity, not sampling, and not the PLS projection. **This finding is not recorded
   anywhere**, and the exit review needs it stated with its numbers.
2. **The localization instrument cannot classify the blocking pair.** `localize_off_diagonal` normalizes
   `delta` by mean path length and `angle` by 180°, but uses `shape` **raw** — a Procrustes distance whose
   whole observed range at this design point is ≈ 0.023 against a null q95 of ≈ 0.018. A single absolute
   threshold of 0.05 therefore exceeds the entire shape response, which is why magnitude/`shape` reports
   `not_material` at every effect while rejecting 74.2 %. The instrument is uninformative for exactly the
   pair that holds the gate.
3. **A size-pure construction may not be buildable here.** The obvious fix — scale all three δ together —
   interacts with per-block standardization (`fit_omics_preprocessor` z-scores each block on pooled data),
   which divides back out part of what the scaling adds; note the joint response above is measured
   *after* standardization, so standardization does not rescue block asymmetry on its own. Whether uniform
   scaling drives the joint `angle`/`shape` to the anchor's exact zero is the one open empirical question.
   If it does not, the correct outcome is a claim revision, and no construction work will produce a clean
   control.

Fact 1 is settled by reading records the repo already has, so this change **records** it rather than
measuring it; only fact 3 needs new computation, and it needs the standardized-geometry checkpoint only —
no RRPP, no PLS fit, no cluster. Fact 2 is a reporting-rule correction. Nothing here changes a gate, a
target, or a committed result.

## What Changes

- **Add a magnitude-construction diagnostic that reads the block decomposition out of existing records.**
  The per-omic scopes are already persisted at `population_standardized` and `observed_standardized`, so
  the probe reports the pooled-versus-per-block response from any merged record set (the Phase 5 run being
  the one that matters) rather than generating new data for it. Its job is to make the finding legible and
  citable, not to discover it.
- **Measure a uniform-δ candidate as a comparator, not as a production mode.** The one open question is
  whether scaling every omic's δ together drives the *joint* `angle`/`shape` to the anchor's zero after
  per-block standardization. Answering it needs only the analytic/standardized geometry checkpoint — no
  RRPP, no PLS fit, no replicate sweep — so it is a small, fast computation. Whether to adopt such a
  `magnitude_kind` is the exit review's decision, and this change deliberately does not make it.
- **Recalibrate localization materiality to each checkpoint's own null dispersion.** Materiality becomes an
  excess expressed in units of the checkpoint's zero-effect null dispersion rather than an absolute
  normalized excess, so the three statistics become commensurable and `shape` stops being structurally
  unclassifiable. The threshold stays descriptive and continues to gate nothing.
- **Confirm the rotation independently with the existing two-stage probe.** `characterize_two_stage`
  (`specificity.py`) gives a shape-free `n_stages=2` configuration. Running magnitude there checks that the
  joint orientation response survives when no shape difference is definable, so the rotation is not an
  artefact of the Procrustes shape-removal step in the four-stage design. This is a cheap independent check
  on an already-established mechanism, not the primary evidence.
- **Write a dated findings report** `docs/reports/magnitude-construction-diagnostic-<date>.md` stating, for
  the exit review: whether block asymmetry explains the response, whether a size-pure construction survives
  standardization, and the recalibrated localization classification for both failing pairs.

**Frozen-result constraint.** The committed Phase 5 `report/` must keep regenerating byte-identically from
`merged.jsonl` — the findings report asserts it does. Changing the localization rule would alter
`phase4_localization.csv`, so the recalibration must not silently change how a previously reported run
renders. Resolving how (opt-in via configuration, versus a recorded re-issue of the affected column) is a
design decision for `design.md`.

Out of scope: adopting a corrected `magnitude_kind` as the production mode; any change to the
`delta`/`angle`/`shape` estimators, RRPP, the generator's sampling model, or the cross-omic coupling; any
change to the Phase 4 gate roles, the acceptance targets, the report contract, or
`phase5_power_study.json`; re-running the paper-grade grid; the Phase 5 exit review's decision itself; the
Phase 6 case study.

## Capabilities

### New Capabilities

- `magnitude-construction-diagnostic`: Decide whether the magnitude surgery's off-target `angle` and
  `shape` response is caused by block-asymmetric δ scaling, and whether a size-pure construction is
  realizable under per-block standardization. Covers the committed probe, its block decomposition, the
  uniform-δ comparator, the shape-free two-stage isolation, and the versioned findings report.

### Modified Capabilities

- `trajectory-power-study`: the localization requirement ("Phase 4 reporting relates operating
  characteristics to realized geometry") changes how a material off-diagonal response is determined —
  materiality is measured in units of each checkpoint's own zero-effect null dispersion, making the
  statistics commensurable, and a previously reported run's localization output must not change silently.

## Impact

- `src/motco/simulations/specificity.py` — probe entry point: block-decomposition reader over merged
  records, the uniform-δ comparator, and reuse of `characterize_two_stage`.
- `src/motco/simulations/study/phase4.py` — `localize_off_diagonal` and `DEFAULT_MATERIALITY_THRESHOLD`
  materiality rule (the substantive code change in this change).
- `src/motco/simulations/semisynthetic.py` — a probe-only uniform-δ candidate path, not a production
  `magnitude_kind`.
- `scripts/` — a driver for the diagnostic. No new study profile is needed: the block decomposition reads
  the committed Phase 5 records, and the comparator needs no replicate sweep.
- `tests/` — tests for the block-decomposition reader, the uniform-δ comparator, and the recalibrated
  materiality rule; existing localization tests updated to the new units.
- `docs/reports/magnitude-construction-diagnostic-<date>.md` — new; `docs/roadmap.md` and
  `docs/phase5-readiness.md` — status pointers.
- `results/phase5-2026-09-10/report/` — must remain byte-identical on regeneration; no re-issue without the
  design decision above.
- No cluster run: the probe is workstation-scale.
