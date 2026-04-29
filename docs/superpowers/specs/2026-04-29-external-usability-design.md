# MOTCO External Usability — Design Spec

**Date:** 2026-04-29
**Scope:** Hardening and small extensions to make MOTCO usable by external scientists
**Primary user:** Scientists/bioinformaticians — both CLI and Python notebook entry points
**Out of scope:** New statistical methods, GUI, web app

---

## Problem Statement

A scientist picking up MOTCO with their own data faces one critical risk: the analysis runs without error but produces wrong results because inputs are misaligned or misspecified. No warning is raised; the science is silently broken. Secondary friction includes zero worked examples, no tests for PLS or SNF, and hidden functionality (VIP scores, configurable spectral components) that would be useful but is unreachable.

---

## Design: Four Ordered Layers

Each layer builds on the previous. Layers 1 and 2 directly eliminate the silent-wrong-results risk. Layers 3 and 4 turn a correct tool into one scientists can actually use with confidence.

---

## Layer 1 — Trust: Input Validation

**Goal:** Any misspecified input raises a clear, descriptive error before computation begins.

### `sd.py`

**`estimate_difference(Y, model_matrix, LS_means, contrast)`**
- Y rows == model_matrix rows; error names both shapes
- LS_means columns == Y columns; error names both
- Every index in every contrast sublist is within `range(len(LS_means))`; error names the offending index
- No NaN or Inf in Y or model_matrix

**`RRPP(Y, model_full, model_reduced, LS_means, contrast, ...)`**
- All of the above, plus:
- model_full rows == model_reduced rows == Y rows

**`get_model_matrix(X, group_col, level_col, full)`**
- group_col and level_col exist in X
- Each column has at least 2 unique values

**`center_matrix(dat, group_col, level_col, feature_cols)`**
- All listed feature_cols exist in dat

### `pls.py`

**`plsda_doubleCV(X, y, ...)`**
- X rows == y rows (or len(y)); error names both
- At least 2 distinct classes in y
- max_components ≤ X.shape[1]; error names both values
- No NaN or Inf in X

### `snf.py`

**`get_affinity_matrix(dats, K, eps)`**
- All datasets have the same number of rows; error names the mismatched shapes
- K < n_samples for every dataset
- eps > 0
- No NaN or Inf in any dataset

**`SNF(Ws, k, t)`**
- All matrices are square
- All matrices have the same shape
- k < n_samples

### `cli.py`

All three subcommands (`plsr`, `snf`, `de`) check shape compatibility of loaded CSVs before calling into stats modules. Error messages become specific: "Y has 100 rows but model_matrix has 50 rows — they must match."

### Error style convention

Every error message must state:
1. The parameter name
2. The observed value
3. What was expected

Example: `ValueError: Y has shape (100, 3) but model_matrix has shape (50, 4) — number of rows must match.`

---

## Layer 2 — Coverage: Tests

**Goal:** Every new validation has a test; every module that currently has zero coverage gets at least smoke-level tests.

### New test files

**`tests/test_validation.py`**
One test per validation rule from Layer 1. Each test:
- Passes a bad input
- Asserts a `ValueError` (or `SystemExit` for CLI paths) is raised
- Asserts the error message contains the relevant context (parameter name, observed value)

**`tests/test_pls.py`**
Smoke tests using a small synthetic dataset (30 samples, 10 features, 2 classes, `random_state=0`):
- `plsda_doubleCV` returns dict with `"table"` and `"models"` keys
- Table has `n_repeats` rows
- All AUROC values in [0, 1]
- Model count matches `n_repeats`
- Validation errors fire correctly for bad inputs

**`tests/test_snf.py`**
Smoke tests using small synthetic arrays (20 samples, 5 features):
- `get_affinity_matrix` returns list of square, symmetric, non-negative matrices
- `SNF` returns square, symmetric matrix of same size
- `get_spectral` returns array of shape (n_samples, n_components)
- Validation errors fire correctly for misaligned inputs and bad K

**`tests/test_cli.py`**
Tests via `motco.cli.main(argv=[...])` — no subprocess needed. Uses `tmp_path` for I/O:
- `plsr` with `--data` + `--label-col`: output table is a valid CSV with correct columns
- `snf` with two synthetic CSVs: fused matrix written; embedding written when `--out-embedding` given
- `de` estimate path: output JSON has `deltas`, `angles`, `shapes` keys with correct shapes
- `de` RRPP path (1 permutation): output JSON has lists of matrices
- Error paths: mismatched row counts, bad contrast JSON, missing required args

### Markers

Add `@pytest.mark.slow` to:
- `tests/test_sd_expected_example1.py::test_example1_expected_results_match`
- `tests/test_sd_expected_example2.py::test_example2_expected_results_match`

Register in `pyproject.toml`:
```toml
[tool.pytest.ini_options]
markers = [
  "slow: marks tests as slow (run with MOTCO_TEST_PERMS=10000)",
]
```

CI fast-test command becomes `pytest -m "not slow"` and actually filters correctly.

---

## Layer 3 — Accessibility: Examples, Docs, CLI Quality-of-Life

**Goal:** A scientist can go from zero to a working analysis using only existing docs and the example, without reading source.

### Example notebook (`examples/motco_example.ipynb`)

Uses `tests/data/evo_649_sm_example1.csv` — runs out of the box, no external data needed.

Sections:
1. **Loading data** — inspect structure; explain what `group_col`, `level_col`, and feature columns mean
2. **Building the model** — `get_model_matrix` and `build_ls_means` with plain-English explanation of dummy coding and the `contrast` parameter
3. **Running `estimate_difference`** — reading the output matrices; what a delta of 0.3 or an angle of 45° means scientifically
4. **Inspecting LS-mean coordinates** — using `get_observed_vectors` to see where each group×level sits before interpreting differences
5. **Running RRPP and computing p-values** — the add-one correction explained; significance at α = 0.05
6. **Equivalent CLI commands** — the three CLI calls that reproduce the notebook results

### README additions

- **"Interpreting your results" section**: plain-English meaning of deltas, angles, and shape distances
  - Angle ≈ 0°: trajectories point in the same direction; Angle ≈ 90°: orthogonal; Angle ≈ 180°: opposite
  - Delta: absolute difference in trajectory magnitude (path length)
  - Shape: Procrustes distance between trajectory shapes after size and orientation are removed
- **"Quick example" section**: three CLI commands that reproduce the notebook, pointing to `examples/`
- Fix placeholder `pyproject.toml` homepage URL

### CLI quality-of-life

- `motco --version`: reads from `__version__` in `src/motco/__init__.py`; implemented via `argparse` `version` action
- `motco de --out-observed PATH`: saves the predicted LS-mean vectors as a CSV (computed as `LS_means @ estimate_betas(model_matrix, Y)` inside `cmd_de`). Note: `get_observed_vectors` is the Python API equivalent for notebook use, but it requires the factor DataFrame and cannot be called directly from CLI inputs.
- All three subcommands: error messages name exact mismatch (row count, column count, bad contrast index)

---

## Layer 4 — Extensions: Exposing Hidden Functionality

**Goal:** Surface what is already implemented but locked away.

### VIP scores (`pls.py`)

- Rename `_calculate_vips` → `calculate_vips` (make public)
- Export from `motco.stats.pls`
- Add `--out-vips PATH` to `motco plsr`: after fitting, computes VIP scores from the best model of each repeat and saves as CSV with rows = features (named from X column headers) and one column per repeat named `rep_1`, `rep_2`, …
- Add to example notebook: VIP scores as a feature-ranking output after `plsda_doubleCV`

### Configurable spectral embedding (`snf.py`)

- Add `n_components: int = 10` parameter to `get_spectral`
- Add `--spectral-components INT` (default 10) to `motco snf`
- Preserves current default behavior; no breaking change

### `get_observed_vectors` exposure

- Already public; no code change needed
- Add to README Python API section and to the example notebook (Layer 3)

### `stats/__init__.py` cleanup

Current `__all__` exports only the string `"stats"`. Add explicit imports:

```python
from motco.stats.sd import (
    estimate_difference, RRPP, get_model_matrix, build_ls_means,
    get_observed_vectors, pair_difference, center_matrix,
)
from motco.stats.pls import plsda_doubleCV, calculate_vips
from motco.stats.snf import get_affinity_matrix, SNF, get_spectral
```

This makes `from motco.stats import estimate_difference` work without knowing the submodule.

---

## File Change Summary

| Layer | Files changed or created |
|-------|--------------------------|
| 1 | `src/motco/stats/sd.py`, `src/motco/stats/pls.py`, `src/motco/stats/snf.py`, `src/motco/cli.py` |
| 2 | `tests/test_validation.py` (new), `tests/test_pls.py` (new), `tests/test_snf.py` (new), `tests/test_cli.py` (new), `tests/test_sd_expected_example1.py`, `tests/test_sd_expected_example2.py`, `pyproject.toml` |
| 3 | `examples/motco_example.ipynb` (new), `README.md`, `pyproject.toml`, `src/motco/cli.py` |
| 4 | `src/motco/stats/pls.py`, `src/motco/stats/snf.py`, `src/motco/stats/__init__.py`, `src/motco/cli.py` |

---

## Non-Goals

- New statistical methods
- Jupyter as a required dependency (notebook is an optional example, not part of the package)
- GUI or web interface
- Performance optimization (benchmarked improvements are deferred; correctness first)
- R parity validation beyond what the existing regression tests already cover
