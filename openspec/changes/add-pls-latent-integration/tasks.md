# Tasks — PLS latent integration

## 1. Implementation
- [x] 1.1 Add `pls` to `IntegrationMethod` literal and to the `_validate_evaluation_params` allow-set in `evaluation.py`.
- [x] 1.2 Implement `_pls_integration(dataset, integration_params)`: standardize-concatenate omic blocks → build stage label Y → `plsda_doubleCV` to select modal LV (parsimony tie-break) → `fit_plsda_transform` for the X-scores → return `LatentIntegrationResult` with `pls_{i}` columns.
- [x] 1.3 Read the stage label from the evaluation params (thread `stage_col` into `_pls_integration`); force `progress=False`; bound `max_components` to the feature count.
- [x] 1.4 Expose CV knobs via `integration_params` (`n_repeats`, `cv1_splits`, `cv2_splits`, `max_components`, `random_state`, `n_jobs`) with documented modest defaults; record the selected LV and AUROC summary in the result metadata.
- [x] 1.5 Dispatch `pls` in `integrate_semisynthetic_dataset`.
- [x] 1.6 Raise a clear `SimulationEvaluationError` when CV is infeasible (too few samples per stage).

## 2. Documentation — latent-space architecture (3 visible homes)
- [x] 2.1 `CLAUDE.md`: add a "Latent-space architecture" block — integration constructs the molecular latent space (the measurement substrate); SNF and PLS are the production latent methods; `concat` is a baseline/diagnostic; viz down-projection is display-only and distinct from measurement.
- [x] 2.2 Spec delta (`specs/simulation-evaluation-harness/spec.md`): MODIFIED integration-methods requirement — add the PLS scenario, mark concat as baseline, state the latent-space-is-measurement-substrate principle.
- [x] 2.3 `evaluation.py` docstrings: module-level + `integrate_semisynthetic_dataset` state the same definition at the point of use.

## 3. Tests
- [x] 3.1 `_pls_integration` returns a `LatentIntegrationResult` with the right shape/columns and `integration_method == "pls"`.
- [x] 3.2 Multi-stage (≥ 3 stages) conditioning works (multiclass one-hot path).
- [x] 3.3 Determinism: same params → identical latent matrix and selected LV.
- [x] 3.4 Selected-LV and CV metadata are recorded in the result.
- [x] 3.5 Validation: unsupported/degenerate cases raise `SimulationEvaluationError`.
- [x] 3.6 End-to-end: a dataset evaluates through `pls` integration to `delta`/`angle`/`shape` (low permutation count).

## 4. Gate
- [x] 4.1 `uv run ruff check src/ tests/` passes.
- [x] 4.2 `uv run mypy src/motco/` passes.
- [x] 4.3 `MOTCO_TEST_PERMS=99 uv run pytest tests/ -m "not slow" --tb=short` passes.
