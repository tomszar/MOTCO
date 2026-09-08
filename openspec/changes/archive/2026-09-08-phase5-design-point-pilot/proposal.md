# phase5-design-point-pilot

## Why

Phase 5 readiness item 4 ([`docs/phase5-readiness.md`](../../../docs/phase5-readiness.md)) owns the 0.80 orientation power floor and the choice of the paper-grade study's design point, and every precondition the geometry audit placed on it has landed: the effect axis is no longer silently censored (P2), the baseline has a declared continuity axis ρ (P4), and each replicate records the configuration eigengap that governs the `angle` null width (P1). What remains is the measurement itself. The Phase 4 pilot's orientation power was flat in requested effect — 0.59, 0.59, 0.64, 0.65 across `e` = 0.25 … 1.00 at n = 300, ρ = 0 — which is the signature of a test limited by null-width dispersion, not by signal. The two levers that shrink that dispersion, samples per group-stage cell and baseline continuity, have never been measured together, and the study machinery cannot cross them: `axes` is strictly one-factor-at-a-time off the baseline, so a config declaring both ρ and `n_samples` enumerates ρ at baseline n and n at baseline ρ but never the (ρ, n) grid the design-point decision needs.

## What Changes

- **Add a crossed design grid to the study configuration.** A new optional `design_grid` block declares one or more namespaced generator/evaluation axes whose values are crossed into design points (columns). Every design point enumerates the full power grid — one mode-agnostic zero-effect anchor plus every (mode × nonzero effect) cell — with the point's coordinates applied to the baseline parameters. The baseline column is required to be present so the existing primary curves are one of the design points. The existing `axes` OFAT block is unchanged and may coexist.
- **Match seeds across the design grid.** Design-grid power cells join the primary matched-seed family, so at a given replicate index every column starts from the same generator seed; because ρ = 0 and ρ > 0 threshold the same uniform block, comparisons along ρ at fixed n are paired at the baseline draw, and the existing distinct-dataset guard still rejects any two cells that would generate identical data. Each column owns its own zero-effect anchor.
- **Enforce headroom per design point.** Enumeration's existing fail-loud surgery-headroom check reads each cell's own ρ and `n_stages`, so it applies per column with no change in policy.
- **Report operating characteristics per design point.** A new `design_point_operating.csv` gives, per (design point, mode, effect, statistic): rejection rate with Monte Carlo SE, the recorded pooled-eigengap distribution, the dispersion of the per-replicate `angle` null width (`q95` median, IQR, SD), and the selected-component distribution — the same covariates the continuity-resolved table already carries, resolved on every design coordinate. The existing continuity-resolved table stops pooling across other design coordinates when a design grid is present, and is byte-identical for studies without one.
- **Predeclare the design-point decision rule in the config and evaluate it in the report.** An `acceptance.design_point` block names the target pair (orientation / `angle`), the power floor (0.80), the confirmation SE threshold, and the column preference order (smallest `n_samples` first, then smallest ρ). The report writes `design_point_decision.json`: the chosen column if any meets the floor with Monte Carlo confirmation, otherwise a `revise_claim` verdict with the continuity-conditional evidence. Non-gating; the verdict is an input to the human decision recorded in the readiness worklist.
- **Commit the pilot profile** `examples/trajectory_power_study/phase5_design_point_pilot.json`: p_dmp = 0.1 (so `e` ≤ 1.00 is realizable for every pool-limited mode at every ρ, including ρ = 0), four stages, `design_grid` = ρ ∈ {0.0, 0.5, 0.8} × `n_samples` ∈ {300, 600, 1200}, modes `orientation` and `translation` (the estimand in question and the headroom-binding negative control; `magnitude` and `shape` reached power 1.00 at n = 300 in Phase 4 and are re-measured in the Phase 5 study at the chosen point), effects 0.25/0.50/1.00, 100 replicates, 199 permutations, matched seeds, attribution off, default `surgery_censoring` (`"error"`). The existing README rule stands: the config does not copy the historical `"clamp"` flag.
- **Run the pilot and record the decision.** A dated findings report under `docs/reports/` ties the verdict to the exact config, code revision, and CSV/JSON outputs; `docs/phase5-readiness.md` item 4 records the chosen design point (or the revised claim) and hands the retained-rank question to item 3.

Out of scope: any change to the statistics, the RRPP test, the surgery transforms, or PLS component selection; varying retained latent rank (item 3, separate change); the Phase 5 paper-grade run itself; group-aware latent-space supervision (cautioned against by the audit).

## Capabilities

### New Capabilities

(none)

### Modified Capabilities

- `trajectory-power-study`: the declarative configuration gains a crossed design grid (with the baseline column required and the axis namespace rules of `axes`); enumeration emits one matched, anchored power grid per design point and applies the headroom check per point; reporting adds a design-point-resolved operating table and a predeclared, config-driven design-point decision; the continuity-resolved view is defined per design point; and the study provides a fixed Phase 5 design-point pilot profile with a versioned findings report.

## Impact

- `src/motco/simulations/study/config.py` — `design_grid` and `acceptance.design_point` parsing/validation; round-trip in `dump_study_config`.
- `src/motco/simulations/study/enumerate.py` — design-point column enumeration (phase `power_design`, metadata carrying the point's coordinates), seed-family assignment, anchor per column; `_generator_identity` already distinguishes ρ and `n_samples`.
- `src/motco/simulations/study/spectrum.py` — design-point-resolved table; continuity table keyed on the full design point.
- `src/motco/simulations/study/report.py`, `targets.py` — new CSV/JSON outputs; design-point decision evaluation; `POWER_PHASES` gains `power_design`. Existing curves, gate, and acceptance targets keep reading the baseline column only.
- `examples/trajectory_power_study/` — new pilot config and README section.
- `scripts/motco_study.py` — no interface change expected; report writes the new files when present.
- Tests — config validation, crossed enumeration (cell count, anchors, seed families, headroom rejection at a non-baseline ρ), report tables, decision rule, backward compatibility of continuity table without a design grid.
- Docs — `docs/phase5-readiness.md` item 4, `docs/roadmap.md` Phase 4→5 hand-off, `CLAUDE.md` study bullet, the dated findings report.
- Compatibility — no generator or evaluation parameter changes, so existing shards and committed results resume and re-report unchanged; the new phase value is additive.
