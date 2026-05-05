## 1. Create New Source Submodules

- [x] 1.1 Create `src/motco/stats/design.py` with `center_matrix`, `get_model_matrix`, `build_ls_means` extracted from `sd.py`
- [x] 1.2 Create `src/motco/stats/trajectory.py` with `pair_difference`, `estimate_difference`, `estimate_betas`, `get_observed_vectors`, and private helpers (`_estimate_size`, `_estimate_orientation`, `_estimate_shape`, `_OPA`) extracted from `sd.py`
- [x] 1.3 Create `src/motco/stats/permutation.py` with `_RRPPWorker` and `RRPP` extracted from `sd.py`

## 2. Update Imports

- [x] 2.1 Update `src/motco/stats/__init__.py` to import from `design`, `trajectory`, and `permutation` instead of `sd`
- [x] 2.2 Update `src/motco/cli.py` to import from `motco.stats` flat namespace instead of `motco.stats.sd`

## 3. Create New Test Files

- [x] 3.1 Create `tests/test_design.py` with tests for `center_matrix`, `get_model_matrix`, `build_ls_means` (from `test_sd_smoke.py` and sd-related tests in `test_validation.py`)
- [x] 3.2 Create `tests/test_trajectory.py` with tests for `pair_difference`, `estimate_difference`, `estimate_betas`, `get_observed_vectors` (from `test_sd_smoke.py` and sd-related tests in `test_validation.py`)
- [x] 3.3 Create `tests/test_permutation.py` with RRPP regression and validation tests (from `test_sd_expected_example1.py`, `test_sd_expected_example2.py`, and sd-related tests in `test_validation.py`)

## 4. Clean Up Old Files

- [x] 4.1 Remove sd-related validation tests from `tests/test_validation.py` (those moved to `test_design.py`, `test_trajectory.py`, `test_permutation.py`)
- [x] 4.2 Update `tests/test_cli.py` import paths
- [x] 4.3 Delete `tests/test_sd_smoke.py`
- [x] 4.4 Delete `tests/test_sd_expected_example1.py`
- [x] 4.5 Delete `tests/test_sd_expected_example2.py`
- [x] 4.6 Delete `src/motco/stats/sd.py`

## 5. Verify

- [x] 5.1 Run `uv run ruff check src/ tests/` — no lint errors
- [x] 5.2 Run `uv run mypy src/motco/` — no type errors
- [x] 5.3 Run `MOTCO_TEST_PERMS=99 uv run pytest tests/ -m "not slow" --tb=short` — all tests pass
