# unify-shape-reflection-policy

## Why

The geometry audit (docs/reports/geometry-audit-2026-09-01.md, finding F3) showed the trajectory
`shape` statistic's documented reflection contract — "reflections retained as distinct shapes" — is
**ambient-dimension-dependent**: whenever the stage configuration is rank-deficient relative to the
ambient space (every pre-integration checkpoint: hundreds of dimensions, configurations of rank ≤ 3),
the determinant correction in `_proper_procrustes_distance` flips a null-space singular vector at zero
cost, so the "proper rotation" constraint is vacuous and the estimator silently returns the
reflection-allowed distance. Only at the `pls_latent` checkpoint (3 dims, rank-3 configurations) does
the constraint bind. Cross-checkpoint shape comparisons — the Phase-4 localization table, and any
Phase-5 report claim that spans checkpoints — therefore mix two different statistics. This is plan
item P3 and must land before Phase-5 report drafting (readiness item 2, Phase-5 report contract).

The audit measured the cost of unification on real replicates: allowing reflections in the latent
space changes the shape distance in **0 of 100** regenerated pilot replicates, and the
reflection-allowed distance still clears the replicate's shape-null q95 in 99 of 100. Reflection is
ruled out as the orientation→shape mechanism; what remains is the rank-limited-projection account,
which this change also instruments (the readiness item-2 follow-up probe).

## What Changes

- **Adopt one reflection policy everywhere: reflections are aligned away** (full orthogonal
  Procrustes alignment, O(k) not SO(k)). Drop the determinant correction from
  `_proper_procrustes_distance` in `src/motco/stats/trajectory.py`. This matches the standard
  identification of reflections in shape spaces whose ambient dimension exceeds configuration rank,
  and makes the statistic identical at every checkpoint by construction. **BREAKING** for the
  statistic's contract in full-rank spaces (a mirror pair now has zero shape distance in 2-D too);
  measured effect on all real study geometry: none.
- Remove the now-dead reflection correction in `_OPA` (unused in production) or align it to the same
  policy, so no second Procrustes convention survives in the module.
- **Add mirror-pair regression tests at ambient dimension > configuration rank** — the regime the
  current suite does not pin (it tests only 2-D full-rank). Under the new policy mirror pairs are
  zero at *every* ambient dimension; the tests pin that rank-independence. Update / replace
  `test_shape_distance_keeps_reflections_distinct_by_default` (tests/test_trajectory.py:299).
- **Guard cross-run comparability**: add a shape-statistic contract version key to the study
  parameter signature (`grid.parameter_signature`) so resuming a pre-change shard under the new
  estimator refuses loudly, mirroring how P1/P2 guarded their schema changes.
- **Item-2 follow-up probe (rank-limited-projection account)**: add a diagnostic
  `forced_components` override to PLS integration (`integration_params`, recorded in integration
  metadata, off by default) and a probe script that re-measures latent shape response in the
  orientation cell at forced ranks above the CV-selected 3. If the orientation→shape response decays
  toward the population value as retained rank grows, the response is predeclared for Phase 5 as a
  projection artifact.
- **Documentation sync**: `trajectory-shape-invariance` spec, README shape description
  (README.md:270), `_estimate_shape` / `_proper_procrustes_distance` docstrings, and the audit's D9
  reflection qualifier (folds into this change).

## Capabilities

### New Capabilities

_None._

### Modified Capabilities

- `trajectory-shape-invariance`: the reflection-policy requirement changes from "documented default:
  reflections retained" to "reflections are aligned away at every ambient dimension" — mirror pairs
  have zero shape distance regardless of ambient dimension, the invariance contract becomes
  rank-independent, and the mirror-pair scenarios are updated accordingly.
- `simulation-evaluation-harness`: gains a diagnostic-only `forced_components` override for PLS
  integration (bypasses double-CV component selection, recorded in integration metadata, default
  absent → behavior unchanged) to support the rank-scaling probe.

## Impact

- `src/motco/stats/trajectory.py` — `_proper_procrustes_distance` (drop determinant branch), `_OPA`
  (remove or align), `_estimate_shape` docstring.
- `src/motco/simulations/evaluation.py` — `_pls_integration` gains the `forced_components` override;
  integration metadata records it.
- `src/motco/simulations/grid.py` — shape-statistic version key in `parameter_signature` (old shards
  correctly refuse to resume; regenerable JSONL is gitignored, committed reports unaffected).
- `tests/test_trajectory.py` — reflection tests replaced; new rank-regime mirror-pair tests.
- `scripts/` — new rank-scaling probe driver (orientation cell, forced ranks, shape response decay).
- Docs: `openspec/specs/trajectory-shape-invariance/spec.md`, `README.md`, docstrings,
  `docs/phase5-readiness.md` item-2 status once the probe runs.
- Not affected: `delta` and `angle` statistics, RRPP machinery, generator semantics, committed
  Phase-4 report outputs (historical records keep their recorded values; comparisons against them
  must note the contract change).
