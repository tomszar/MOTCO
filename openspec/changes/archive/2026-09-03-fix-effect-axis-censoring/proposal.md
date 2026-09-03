# fix-effect-axis-censoring

## Why

The geometry audit ([finding F2](../../../docs/reports/geometry-audit-2026-09-01.md)) showed the
orientation and translation effect axes are silently censored: `_relocate_rows` clamps the relocation
count to the destination pool (`k = min(k, |dst_pool|)`), and the translation surgery clamps its extra
set the same way. With `p_dmp = 0.2` and four stages ~59% of CpGs are stage-active, so the orientation
surgery saturates at e ≈ 0.69 — **80 of 100 matched-seed replicate pairs at e = 0.75 and e = 1.00 are
byte-identical datasets** reported as independent power measurements. The study's existing
distinct-dataset guard compares *parameters*, so it cannot catch this: the parameters differ, the
realized surgery does not. Readiness item 4 (the Phase-5 design-point study) is preconditioned on
fixing this axis — running a power grid on the current axis re-measures the same data in its top cells.

## What Changes

- **BREAKING** — The semisynthetic generator gains an explicit censoring policy for pool-limited
  surgeries (orientation, translation, shape/relocate). Default `"error"`: generation fails loudly
  when the clamp would bind (the requested surgery cannot be realized in full). Explicit opt-in
  `"clamp"` preserves the old behavior for exploratory use. The new parameter enters
  `SemiSyntheticTrajectoryParams`, so grid parameter signatures change and old shards will
  (correctly) refuse to resume.
- Truth metadata records the **nominal** (requested) surgery size beside the already-recorded
  realized size, plus a per-record `censored` flag, so requested-vs-realized is explicit in every
  replicate record.
- `summarize_rejection_rates`-level study summaries and the study report surface realized surgery
  per cell (mean realized size, nominal size, censored fraction), and the report annotates power
  cells whose realized construction is identical in distribution instead of presenting them as
  independent measurements.
- Study enumeration gains a config-time headroom check: cells whose requested effect exceeds the
  expected destination-pool headroom (in expectation, from `p_dmp` and `n_stages`) are rejected at
  enumeration time — before any cluster compute is spent — with a message naming the offending cell
  and the saturating fraction.
- Non-goal: no change to any statistic, test, or the surgery mechanics themselves when the clamp
  does not bind. Re-running the Phase-4 grid is out of scope (that is readiness item 4's design-point
  study, which this unblocks).

## Capabilities

### New Capabilities

(none)

### Modified Capabilities

- `semisynthetic-trajectory-generator`: pool-limited surgeries must apply an explicit censoring
  policy (error by default, clamp by opt-in) and record nominal vs realized surgery sizes and a
  censored flag in truth metadata.
- `simulation-grid-orchestration`: cell summaries must surface realized surgery (nominal size,
  realized size, censored fraction) beside rejection rates.
- `trajectory-power-study`: study enumeration must reject cells whose requested effect exceeds the
  expected surgery headroom; study reports must annotate realized effects per cell and flag cells
  whose realized construction is identical in distribution.

## Impact

- `src/motco/simulations/semisynthetic.py` — `SemiSyntheticTrajectoryParams` (new field),
  `_relocate_rows`, `_orientation_methyl`, `_shape_methyl`, `_translation_methyl`, truth assembly.
- `src/motco/simulations/grid.py` — no schema change needed for records (`truth_metadata` already
  flows through), but summary surfacing; parameter signatures change via the new params field.
- `src/motco/simulations/study/enumerate.py` — headroom validation beside
  `_require_distinct_primary_datasets`.
- `src/motco/simulations/study/summary.py`, `study/report.py` — realized-surgery table and cell
  annotations.
- `examples/trajectory_power_study/*.json` — existing configs with censored cells (the Phase-4
  pilot's orientation e ∈ {0.75, 1.00} and translation top cells) now fail enumeration by design;
  they stay as the historical record and any future grid must choose an uncensored axis.
- Tests: generator censoring-policy tests, enumeration headroom tests, summary/report surfacing
  tests; existing generator tests that exercise saturating effects will need explicit
  `"clamp"` or lower effects.
- Acceptance (from the audit): enumerating the Phase-4 grid under the new policy errors on the
  censored cells (or, under `"clamp"`, reports distinct realized effects per cell); no two power
  cells share > 5% identical replicates.
