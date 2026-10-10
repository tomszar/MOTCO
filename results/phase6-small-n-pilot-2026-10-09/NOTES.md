# Phase 6 small-n pilot — notes (2026-10-09)

Pilot of `examples/trajectory_power_study/phase6_small_n_pilot.json` at the SEA-AD MTG astrocyte design:
three stages, the ≥30-nuclei donor table F 11/10/28, M 9/10/12 (n = 80; group A = F, group B = M),
methylation + expression measured (`integration_params.layers`), pooled PLS with the stage-supervised
double-CV rank, `magnitude_kind = joint`. Baseline column only, 100 replicates × 199 permutations, gate
disabled, shared axis 0 / 0.005 / 0.01 / 0.02 / 0.05 / 0.10 / 0.25 / 0.50 / 1.00 for all four modes.
3,500 units, 0 failures, 0 censored surgeries (`PROVENANCE.txt`). Rates are from `report/`; realized geometry,
null quantiles, eigengaps and ranks from `merged.jsonl` (gitignored; sha256 in `PROVENANCE.txt`). At 100
replicates the Monte Carlo SE of a rate p is √(p(1−p)/100): 0.022 at 0.05, 0.05 at 0.5.

## 1. Type I is controlled

`type_i_baseline` `none`: `delta` 0.05, `angle` 0.01, `shape` 0.01. The translation negative control at
e = 1.00: 0.02 / 0.00 / 0.00. The shared zero-effect anchor: 0.03 / 0.03 / 0.02. Translation as a power mode
(`report/power_curves.csv`) stays at 0.00–0.05 on every statistic at every effect. None of these exceeds
α + 2·SE (0.094 at 100 replicates).

## 2. Target-statistic power by mode

| e | magnitude / `delta` | orientation / `angle` | shape / `shape` |
|---|---|---|---|
| 0 (anchor) | 0.03 | 0.03 | 0.02 |
| 0.005 | 0.05 | 0.04 | 0.02 |
| 0.01 | 0.04 | 0.04 | 0.02 |
| 0.02 | 0.09 | 0.01 | 0.03 |
| 0.05 | 0.17 | 0.05 | 0.03 |
| 0.10 | 0.58 | 0.05 | 0.03 |
| 0.25 | 0.93 | 0.01 | 0.07 |
| 0.50 | 1.00 | 0.05 | 0.09 |
| 1.00 | 1.00 | 0.24 | 0.18 |

**Magnitude** rises between e = 0.02 and 0.25 and saturates at 0.50. The rise sits about 10× higher than at
n = 1200 (Phase 5 re-measurement: power 0.84 at e = 0.01, 1.00 at 0.02). The realized PLS-latent `delta` grows
as expected (median 1.08 / 1.93 / 3.69 / 8.27 at e = 0.02 / 0.05 / 0.10 / 0.25), but the `delta` null's
95th percentile is about 3.4–3.5, against 0.40 at n = 1200. The `delta` null widens roughly as 1/√n
inflated by the smaller, unbalanced cells, which moves the rise. Both controls stay at the floor:
magnitude/`angle` 0.01–0.03 and magnitude/`shape` 0.00–0.02 at every effect.

**Orientation and shape do not rise within the construction's range.** Both modes relocate a fraction e of
methylation sites, and `_relocate_rows` clamps that fraction at 1.0, so e = 1.00 is the strongest orientation
and shape construction the generator can produce; there is no larger effect to add. At that maximum,
orientation/`angle` reaches 0.24 and shape/`shape` 0.18.

The records explain why:

- **The `angle` null is very wide at three stages.** With three stage means the configuration is a triangle
  in a 2-D subspace, and its pooled relative eigengap is small: anchor median 0.109 (deciles 0.039 / 0.221).
  PC1 is then poorly determined, and the `angle` null q95 is 43–138° across cells (anchor median 69°).
- **PLS attenuates the rotation.** The population-standardized joint `angle` of the orientation construction
  is 48° at e = 0.25 and 90° at e = 1.00, but the median PLS-latent `angle` is 20° and 67°. The
  stage-supervised latent space keeps the stage axis that both groups share and drops much of the
  group-specific rotation.
- **The shape construction is small next to the null.** Population-standardized `shape` at e = 1.00 is 0.046,
  the PLS-latent median 0.062, and the `shape` null q95 0.105.

Orientation at e = 1.00 also moves the other two statistics more than its own (`delta` 0.33, `shape` 0.43
vs `angle` 0.24), and the CV rank there spreads to 4–9 in 51 of 100 replicates (elsewhere it is 2 or 3 in
every replicate). Shape at e = 1.00: `delta` 0.28, `angle` 0.12, `shape` 0.18.

## 3. Decision: split the paper-grade axis into two profiles

The spec's split rule applies: no single axis resolves every mode's target-statistic rise. Magnitude's rise
lies in 0.02–0.25. Orientation's and shape's only measurable movement lies in 0.50–1.00, at the
construction's ceiling, where they never reach the 0.80 floor. One shared axis would either leave
magnitude's rise unresolved or spend most of the orientation/shape cells at points the pilot measured at
the null.

- **`phase6_small_n_magnitude.json`** — magnitude only, effects **0 / 0.02 / 0.05 / 0.10 / 0.25 / 0.50 /
  1.00**. 0.02–0.25 trace the rise (0.09 / 0.17 / 0.58 / 0.93); 0.50 is its saturation; 1.00 is the control
  stress point. The two smallest pilot points (0.005, 0.01) sat at the anchor's rate and are dropped.
- **`phase6_small_n_study.json`** — orientation, shape and translation, effects **0 / 0.25 / 0.50 / 0.75 /
  1.00** (the Phase 5 axis). 1.00 is the construction maximum and the pilot's only point clearly above α;
  0.75 is added between the pilot's 0.50 and 1.00 to resolve the curve where it starts to move; 0.25 is kept
  as the point where the realized population rotation is already large (48°) but the measured rate is
  null-like, which is itself a finding for the case study. Points below 0.25 measured at the null for both
  modes and are dropped.

Both profiles keep the pilot's `generator`, `evaluation`, base seed (800) and matched-seed family
(`phase6-small-n`). Their shared zero-effect anchors are therefore the same datasets, and the baseline
column's first 100 replicates reproduce this pilot's records at the same replicate indices.

**Columns.** Both profiles cross `generator.group_stage_sizes` (≥30 nuclei, ≥50 nuclei) with
`evaluation.integration_params.layers` (methylation + expression, all three) in the design grid. The engine
crosses design-grid axes, so this gives four columns instead of design D4's three. The fourth (≥50 nuclei
on three blocks) was accepted (2026-10-09) rather than adding an explicit-points mode to the engine. The
cohort-size contrast is still read baseline vs ≥50 nuclei at two blocks, and the block-count contrast
baseline vs three blocks at n = 80. The fourth column adds the interaction.

**Not changed by the pilot.** The 0.80 power floor is not expected to be met for orientation or shape at
n = 80. Per design D6 that is the finding the paper-grade run will measure with 500 × 999 precision, not a
defect to engineer around. Raising `baseline_continuity` (larger eigengap) or `p_dmp` would raise
orientation power, but it would no longer describe this cohort's design. Whether the case study can read
`angle` at all depends on the SEA-AD cohort's own recorded eigengap, which the report must stratify by.

## 4. Cost for the paper-grade run

Pilot median 8.0 s per unit at 199 permutations (8.59 core-hours for 3,500 units). A local timing at
999 permutations, attribution on, gave 7–16 s per unit across the four columns. Attribution (100 bootstraps)
adds under 0.5 s at n = 80. Planning figure: 10 s per unit, plus 20% for node contention.

| Profile | Cells | Units (× 500) | Core-hours at 10 s | with +20% |
|---|---|---|---|---|
| magnitude | 2 Type I + 4 columns × (1 anchor + 6) = 30 | 15,000 | 41.7 | 50 |
| orientation / shape / translation | 2 Type I + 4 columns × (1 anchor + 3 × 4) = 54 | 27,000 | 75.0 | 90 |
| total | 84 | 42,000 | 116.7 | 140 |
