# Phase 5 magnitude re-measurement pilot — notes (2026-10-08)

Pilot of `examples/trajectory_power_study/phase5_magnitude_pilot.json`: the magnitude mode under the
`joint` construction (`magnitude_kind = "joint"`, every omic's δ scaled by `1 + e`) at the Phase 5
paper-grade design point, 50 replicates × 199 permutations, gate disabled. Two submissions into this run
directory (`PROVENANCE.txt`): submission 1 on the bracket-chosen grid 0 / 0.02 / 0.05 / 0.10 / 0.25 / 1.00,
submission 2 extending it with 0.0025 / 0.005 / 0.01. 550 units, 0 failures, 0 censored surgeries. Every
number below is in `report/` or `merged.jsonl` (gitignored; sha256 in `PROVENANCE.txt`).

## 1. The bracket misplaced the rise; the pilot found it

Design D5 chose the pilot grid from the analytic bracket so that joint's realized population size at the
sub-0.25 points straddled the realized size (5.95) at which the `all` construction's Phase 5 `delta` power
is already 1.000. Submission 1 measured `delta` power **1.000 at every nonzero effect, including e = 0.02**
(`report/power_curves.csv`): the population bracket relates the two constructions' sizes but cannot see the
RRPP null width, which is the sampled quantity the pilot adds — exactly the risk D5 recorded.

The pilot's own records locate the rise. In the PLS latent space the realized joint `delta` is linear in e at
small e (median 1.38 at e = 0.02, 3.39 at 0.05, 6.58 at 0.10 — about 68 × e; `realized_geometry`,
`pls_latent`/`joint`), while the `delta` null's 95th percentile is about 0.40 at every cell
(`null_summary.delta.q95`, median 0.39–0.47 up to e = 0.10). Power 0.5 was therefore expected near e ≈ 0.006.
Submission 2 added 0.0025 / 0.005 / 0.01 (the bracket regenerated at step 0.0025 so the points are on the
committed CSV: realized population size 0.164 / 0.328 / 0.655).

| e | realized joint `delta`, population-standardized (bracket) | realized joint `delta`, PLS latent (pilot median) | `delta` null q95 (median) | `delta` power ± MC SE | `angle` rate | `shape` rate |
|---|---|---|---|---|---|---|
| 0 (anchor) | 0 | 0.158 | 0.387 | 0.04 ± 0.028 | 0.02 | 0.10 |
| 0.0025 | 0.164 | 0.233 | 0.395 | 0.22 ± 0.059 | 0.02 | 0.10 |
| 0.005 | 0.328 | 0.389 | 0.394 | 0.46 ± 0.070 | 0.02 | 0.10 |
| 0.01 | 0.655 | 0.722 | 0.399 | 0.84 ± 0.052 | 0.02 | 0.10 |
| 0.02 | 1.306 | 1.376 | 0.405 | 1.00 | 0.02 | 0.10 |
| 0.05 | 3.229 | 3.388 | 0.424 | 1.00 | 0.02 | 0.04 |
| 0.10 | 6.340 | 6.580 | 0.470 | 1.00 | 0.02 | 0.02 |
| 0.25 | 14.962 | 15.377 | 0.735 | 1.00 | 0.02 | 0.00 |
| 1.00 | 44.798 | 46.232 | 1.889 | 1.00 | 0.00 | 0.00 |

(`report/power_curves.csv` for the rates; `merged.jsonl` for the realized geometry and null quantiles; the
anchor row pools the shared zero-effect anchor's 50 records, `report/type_i_table.csv` carries the separate
`type_i_baseline` cells.) The `delta` curve rises monotonically across 0.0025 → 0.02 and saturates from 0.02.

## 2. Controls sit at α at every effect

`magnitude`/`angle` is 0.02 at every nonzero effect (0.00 at e = 1.00) and `magnitude`/`shape` is 0.10 at the
four smallest effects, falling to 0.00 by e = 0.25 — against the anchor's own 0.02 and 0.10 at 50 replicates.
The α + 2·SE bound for a rate of 0.10 at 50 replicates is 0.135, so no control is outside it; the shape rate
at small e equals the anchor's and shows no trend with e. (The anchor's 0.10 shape rate at 50 replicates is
Monte Carlo noise around the 0.05 nominal — its MC SE is 0.042 — and is the same anchor Phase 5 measured
at 0.046 over 500 replicates.) Compare Phase 5's `all` construction at e = 1.00: `angle` 0.276, `shape`
0.742. The realized PLS-latent `angle` under `joint` is 1.6–2.0° at every effect (anchor 1.96°), against a
null q95 of 5.3–27°, and `shape` 0.005 (anchor 0.005). The pilot does not gate; the paper-grade run does.

## 3. Anchor reproduction

`report/anchor_reproduction.csv` (`scripts/anchor_reproduction.py`): at all 50 replicate indices the pilot's
anchor and the Phase 5 paper-grade anchor (`power_primary-d74045b65506`) have identical generator seeds,
identical selected PLS rank, and observed `delta`/`angle`/`shape` equal to within 6e-12 absolute. The p-values
are not comparable (199 vs 999 permutations; mean absolute difference 0.016–0.021) and are excluded from the
verdict; the paper-grade run at 999 permutations is expected to reproduce them exactly.

## 4. Selected PLS rank

`integration_metadata.selected_lv` is 3 at 91–98 % of replicates in every cell (2 otherwise; never above 3),
with no trend in e: 46/50 at e = 0.0025 through 0.02, 47/50 at 0.25, 49/50 at 1.00, 91/100 across the two
`none` cells. Small e does not move CV rank selection away from `n_stages − 1`.

## 5. Decision: the paper-grade grid differs from the pilot's submission-1 grid

`phase5_magnitude_remeasurement.json` uses **0 / 0.0025 / 0.005 / 0.01 / 0.02 / 0.25 / 1.00**, not the
bracket-chosen 0 / 0.02 / 0.05 / 0.10 / 0.25 / 1.00. Reason, per design D5: the pilot showed the rise was
misplaced (above), and every paper-grade point is one the pilot measured. 0.0025 / 0.005 / 0.01 bracket the
rise (0.22 / 0.46 / 0.84 at 50 replicates), 0.02 is its saturation, 0.25 is kept because it is the `all`
construction's first Phase 5 point (joint realizes 14.96 there against `all`'s 5.95 — the axis relation the
addendum must state), and 1.00 is the stress point for the two controls (2.47 × the production construction's
realized size). 0.05 and 0.10 are dropped: both measured at power 1.000 with controls at the floor, so they
add nothing the neighbouring points do not already say. Cost: 9 cells × 500 = 4,500 units.
