## 1. Unify the reflection policy in the estimator

- [x] 1.1 Drop the determinant correction in `_proper_procrustes_distance` (`src/motco/stats/trajectory.py`) so the alignment is plain orthogonal Procrustes; add `SHAPE_STATISTIC_VERSION = 2` beside it; update the `_estimate_shape` and `_proper_procrustes_distance` docstrings to state the reflections-aligned-away policy; verify `uv run pytest tests/test_trajectory.py -v` runs (reflection tests updated in 1.3).
- [x] 1.2 Delete the unused `_OPA` helper and verify no references remain: `grep -rn "_OPA" src/ tests/ scripts/` returns nothing and mypy passes (`uv run mypy src/motco/`).
- [x] 1.3 Replace `test_shape_distance_keeps_reflections_distinct_by_default` with mirror-pair tests pinning the new contract: mirror pair zero at full rank (2-D), zero at ambient > rank (the audit's 3/10/660-dim embeddings of the bent planar trajectory), and shape distance invariant to zero-padding embedding dimension; verify with `uv run pytest tests/test_trajectory.py -v`.
- [x] 1.4 Add the no-op regression test (design D4): on random full-rank 3-D configurations, old-proper and new-orthogonal distances agree whenever the optimal alignment is already proper; verify with `uv run pytest tests/test_trajectory.py -v`.

## 2. Guard cross-run comparability

- [x] 2.1 Add `"shape_statistic_version": SHAPE_STATISTIC_VERSION` (imported from `motco.stats.trajectory`) to the `parameter_signature` payload in `src/motco/simulations/grid.py`, extending the existing schema-version comment block; verify with a unit test asserting the signature changes when the constant changes and that resuming a shard written under a different signature is refused (existing mismatch path).

## 3. Forced-component override for PLS integration

- [x] 3.1 Implement `integration_params["forced_components"]` in `_pls_integration` (`src/motco/simulations/evaluation.py`): validate feasibility (reject, not clamp, when outside 2..min(n_features, n_samples)), skip `plsda_doubleCV`, fit the single pooled model at the forced count, and record `forced_components` plus `component_selection: "forced"` (vs `"cv"` on the default path) in integration metadata; verify with new tests in `tests/test_simulation_evaluation.py` covering the three spec scenarios (default unchanged with `component_selection == "cv"`, forced honored and recorded, infeasible rejected).
- [x] 3.2 Add the study-report guard asserting production records carry `component_selection == "cv"` (design D5) and verify with a unit test feeding a forced-rank record.

## 4. Rank-scaling probe (readiness item-2 follow-up)

- [x] 4.1 Write `scripts/latent_rank_probe.py` following `scripts/geometry_specificity_probe.py` conventions: regenerate orientation-cell replicates from matched seeds, evaluate at forced ranks (default ladder 3, 4, 6, 9, 12, clamped to feasibility), and emit per-rank observed `shape`, RRPP rejection against each replicate's own null, and the pooled `config_spectrum` eigengap as CSV + markdown summary under `results/`; verify the script runs end-to-end on a reduced smoke configuration (few replicates, low permutations).
- [x] 4.2 Run the probe at the pilot design point, commit the summary outputs under `results/`, and record whether the orientation→shape response decays toward the population value as rank grows; verify the summary states the decay measurement and its direction.

## 5. Documentation sync

- [x] 5.1 Update `README.md` shape description (line ~270) and any remaining "reflections retained" wording (audit D9): `grep -rn "reflection" README.md docs/ src/motco/` shows only the new policy; verify the pre-commit gate passes (`uv run ruff check src/ tests/ && uv run mypy src/motco/ && MOTCO_TEST_PERMS=99 uv run pytest tests/ -m "not slow" --tb=short`).
- [x] 5.2 Update `docs/phase5-readiness.md` item 2 with the P3 resolution and probe outcome, marking the reflection-policy precondition closed; verify the item references the probe results committed in 4.2.
