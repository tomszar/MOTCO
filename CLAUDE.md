# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (lockfile-respecting, matches CI)
uv venv && source .venv/bin/activate
uv sync --extra test   # or: uv pip install -e ".[test]"

# Lint
uv run ruff check src/ tests/

# Type check
uv run mypy src/motco/

# Run fast tests (no pytest slow markers exist; speed is controlled by env var only)
MOTCO_TEST_PERMS=99 uv run pytest tests/ --tb=short

# Run a single test file
uv run pytest tests/test_sd_smoke.py -v

# Run slow regression tests with parallelism (only on main in CI)
MOTCO_TEST_PERMS=10000 MOTCO_N_JOBS=-1 uv run pytest tests/ --tb=short

# CLI entry point
motco --help
motco plsr --help
motco snf --help
motco de --help
```

## Architecture

MOTCO is a Python package (`src/motco/`) with a CLI entry point (`motco`) and three statistical modules under `src/motco/stats/`:

- **`stats/pls.py`** — PLS-DA with double cross-validation (`plsda_doubleCV`). The outer loop (CV2) picks the best model per repeat; the inner loop (CV1) selects the number of latent variables by AUROC. Returns a dict with `"table"` (DataFrame) and `"models"` (list of fitted `PLSRegression`). Optional parallelism via `n_jobs`.

- **`stats/snf.py`** — Similarity Network Fusion. `get_affinity_matrix` → `SNF` → optional `get_spectral`. Input datasets must be sample-aligned (same rows, same order). Spectral embedding is fixed at 10 components.

- **`stats/sd.py`** — Trajectory analysis (group differences). Core pipeline:
  1. `get_model_matrix(X, group_col, level_col, full)` — builds an intercept + dummy-coded design matrix with optional group×level interactions. Category order is deterministic (sorted string representation).
  2. `build_ls_means(group_levels, level_levels, full)` — generates the LS-mean row vectors consistent with the coding from `get_model_matrix`. Row order is group-major, level-minor.
  3. `estimate_difference(Y, model_matrix, LS_means, contrast)` — fits betas via normal equations (Cholesky → direct solve → lstsq fallback), then computes per-group trajectory size (`_estimate_size`), orientation (`_estimate_orientation` via eigendecomposition), and shape (`_estimate_shape` via iterative Procrustes GPA). Returns symmetric matrices `(deltas, angles, shapes)`.
  4. `RRPP(Y, model_full, model_reduced, LS_means, contrast, permutations, n_jobs)` — permutes residuals of the reduced model and calls `estimate_difference` for each permutation. Serial by default; parallel via `multiprocessing.Pool` using `_RRPPWorker` (a picklable callable).

### Key conventions

- All model matrices must include an intercept column. `get_model_matrix` always prepends one.
- `contrast` is a `list[list[int]]`: each inner list enumerates the LS-mean row indices belonging to one group (group-major, level-minor indexing matches `build_ls_means`).
- `group_col` and `level_col` must always be passed explicitly — no defaults anywhere.
- `MOTCO_TEST_PERMS` controls permutation count in regression tests (default `10000`). `MOTCO_N_JOBS` controls RRPP parallelism (default `1`; use `-1` for all CPUs).
- Regression test reference values are in `tests/data/results_example1.csv` and `tests/data/results_example2.csv`. `tests/data/reference/` contains the R script used to generate them, not CSV outputs.
- The package is built with `hatchling`; version is sourced from `src/motco/__init__.py`.
