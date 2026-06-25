# Tasks — Rung 3: cross-talk through the production latent spaces

## 1. Instrumentation
- [x] 1.1 Add `integration_method` + `integration_params` parameters (default `concat`, `None`) to `evaluate_mode_specificity`; forward them to the RRPP `evaluate_semisynthetic_trajectory` call.
- [x] 1.2 Forward the same selection to the group-in-stage projection (`_group_in_stage_fraction` via its `SimulationEvaluationParams`), so the projection is measured in the selected latent space.
- [x] 1.3 Pass-through `integration_method`/`integration_params` in `characterize_two_stage`.
- [x] 1.4 Record the effective integration method (and PLS selected-LV / CV params where applicable) in the returned/printed summary.

## 2. Driver
- [x] 2.1 Add `scripts/latent_space_crosstalk_probe.py`: run each mode through `concat`/`snf`/`pls` on matched seeds + effect size, with modest PLS CV knobs.
- [x] 2.2 Print the per-statistic rejection-rate table and group-in-stage fraction per latent space, in a findings-ready layout.

## 3. Tests
- [x] 3.1 `evaluate_mode_specificity` runs through `pls` and records the integration method (fast: tiny replicate/permutation/CV counts).
- [x] 3.2 Backward compatibility: the default path is still `concat` (existing behavior unchanged).
- [x] 3.3 `characterize_two_stage` forwards the latent-space selection.

## 4. Findings
- [x] 4.1 Run the driver; capture per-latent-space rejection-rate tables.
- [x] 4.2 Write `findings.md`: does cross-talk reproduce through PLS? confirm/refute Rung 2's "clean-null but lossy" prediction on the real generator; gate decision for the next rung.

## 5. Gate
- [x] 5.1 `ruff check src/ tests/` passes.
- [x] 5.2 `mypy src/motco/` passes.
- [x] 5.3 `MOTCO_TEST_PERMS=99 pytest tests/ -m "not slow"` passes.
