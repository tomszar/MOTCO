## 1. Core API — fit_plsda_transform

- [x] 1.1 Add `fit_plsda_transform(X, y, n_components) -> np.ndarray` to `src/motco/stats/pls.py`: fit `PLSRegression(n_components, scale=True)` on full X/y and return `x_scores_`
- [x] 1.2 Export `fit_plsda_transform` from `src/motco/stats/__init__.py`
- [x] 1.3 Write unit tests for `fit_plsda_transform`: correct output shape, values match sklearn directly, type check

## 2. CLI — Multi-omics input mode

- [x] 2.1 Add `--input` (repeatable `action="append"`) to the `plsr` subparser in `build_parser`
- [x] 2.2 Add `--metadata` (CSV path) and `--label-col` (column name) arguments to the `plsr` subparser
- [x] 2.3 Add input validation in `cmd_plsr`: `--input` and `--data`/`--x`/`--y` are mutually exclusive; `--input` requires both `--metadata` and `--label-col`; label column must exist in metadata CSV
- [x] 2.4 Implement multi-omics loading in `cmd_plsr`: read each `--input` file, standardize per layer (z-score; std=0 → 1), concatenate horizontally to form X; read `--label-col` from `--metadata` as y

## 3. CLI — Score output

- [x] 3.1 Add `--out-scores` (path, optional) and `--n-components` (int, optional) to the `plsr` subparser
- [x] 3.2 After double-CV completes, if `--out-scores` is set: compute modal LV from `res["table"]["LV"]` (or use `--n-components` if provided), call `fit_plsda_transform`, save scores as CSV with columns `lv_1, lv_2, …`
- [x] 3.3 Ensure `--out-scores` works with both single-matrix mode (`--data`/`--x`/`--y`) and multi-omics mode (`--input`)

## 4. Tests

- [x] 4.1 CLI integration test: `--input` with two small CSVs + `--metadata` + `--label-col` runs without error and produces a valid CV table
- [x] 4.2 CLI integration test: `--out-scores` produces a CSV with the correct number of rows and columns (modal LV count)
- [x] 4.3 CLI integration test: `--n-components 2` overrides modal LV and produces a score matrix with 2 columns
- [x] 4.4 CLI validation tests: `--input` + `--data` raises SystemExit; `--input` without `--metadata` raises SystemExit; missing label column raises SystemExit

## 5. Pre-commit gate

- [x] 5.1 Run `uv run ruff check src/ tests/` — no errors
- [x] 5.2 Run `uv run mypy src/motco/` — no new type errors
- [x] 5.3 Run `MOTCO_TEST_PERMS=99 uv run pytest tests/ -m "not slow" --tb=short` — all pass
