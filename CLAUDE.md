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

# Run fast tests
MOTCO_TEST_PERMS=99 uv run pytest tests/ -m "not slow" --tb=short

# Run a single test file
uv run pytest tests/test_trajectory.py -v

# Run slow regression tests with parallelism (only on main in CI)
MOTCO_TEST_PERMS=10000 MOTCO_N_JOBS=-1 uv run pytest tests/ --tb=short

# Pre-commit gate — all three must pass before committing
uv run ruff check src/ tests/ && uv run mypy src/motco/ && MOTCO_TEST_PERMS=99 uv run pytest tests/ -m "not slow" --tb=short

# CLI entry point
motco --help
motco plsr --help
motco snf --help
motco simulate --help   # numpy generator (no R at runtime)
motco de --help
```

## Architecture

MOTCO is a Python package (`src/motco/`) with a CLI entry point (`motco`) and statistical modules under `src/motco/stats/`:

- **`stats/pls.py`** — PLS-DA with double cross-validation (`plsda_doubleCV`). The outer loop (CV2) picks the best model per repeat; the inner loop (CV1) selects the number of latent variables by AUROC. Returns a dict with `"table"` (DataFrame) and `"models"` (list of fitted `PLSRegression`). Optional parallelism via `n_jobs`. `calculate_vips(model)` computes VIP scores from a fitted model.

- **`stats/snf.py`** — Similarity Network Fusion. `get_affinity_matrix` → `SNF` → optional `get_spectral(n_components=10)`. Input datasets must be sample-aligned (same rows, same order).

- **`cli.py`** — Argparse CLI with four subcommands: `plsr`, `snf`, `simulate` (semi-synthetic multi-omic dataset via the numpy generator, no R), `de`. Each wraps the corresponding module and converts `ValueError` to clean `SystemExit` messages.

- **`viz.py`** — Visualization. Two-layer matplotlib API for trajectory geometry:
  - `plot_trajectories(observed_vectors, projector, ...)` — core function; takes pre-computed LS-mean vectors (output of `get_observed_vectors`) and a fitted projector, draws directed 2D paths with per-segment direction arrows.
  - `plot_trajectory_from_data(Y, metadata, group_col, level_col, ...)` — convenience wrapper; fits PCA on `Y` and returns `(fig, ax, pca)` so the projector can be reused.

- **`simulations/`** — Semi-synthetic simulation framework (numpy-native; no R at runtime):
  - `reference.py` — Loads the cached InterSIM reference data (`data/intersim_reference.npz`: per-omic means/covariances, cross-omic incidence maps, correlation vectors) with `load_reference()`. The cache is produced once by `export_reference.R` (run in R) + `build_cache_from_export(...)`; runtime never touches R.
  - `generator.py` — numpy reimplementation of InterSIM's generative model (`μ = base + δ·v` → `MVN(μ, Σ)`, methylation `rev.logit` after the M-value shift, cross-omic coupling via the incidence maps). `generate_omics(...)` samples per-cell from explicit differential indicators and returns matrices + indicator truth. Realism is validated against an InterSIM fixture in `tests/test_generator.py`.
  - `intersim.py` — Legacy bridge to the R InterSIM package, retained **only** for the one-time reference-data export; not used in runtime generation.
  - `fidelity.py` — Paper-grade fidelity validation of the numpy generator against InterSIM: a per-omic statistic battery (`compute_statistics`), a `delta × p.DMP` grid (`default_grid`), a numpy cell runner over the real generator (`run_numpy_cell`), and the replicate-distribution criterion (`compare_cell`/`validate_grid` — numpy's replicate mean must fall in InterSIM's `[q2.5, q97.5]` interval). The InterSIM side is committed R-free as `tests/data/intersim_fidelity_fixture.npz`; `fidelity_intersim.R` regenerates it and `build_fidelity_fixture_from_export` packs it. Tests in `tests/test_generator_fidelity.py`; supplement via `scripts/fidelity_supplement.py`. See `simulations/FIDELITY.md` for the full protocol and reproduction steps.
  - `fidelity_visual.py` — Qualitative side-by-side InterSIM-vs-numpy figures (density, per-modality clustermap heatmaps with sample/feature dendrograms + cluster colour bar, per-modality PCA at 4 clusters, per-feature moment-agreement scatter, cross-omic coupling correlation block). Unlike the quantitative fixture, the InterSIM **raw matrices are not committed** (large; only needed to render the supplement) — regenerate them with InterSIM via `flake.nix` using `fidelity_visual_intersim.R` + `build_visual_fixture_from_export` (writes under gitignored `build/`); the numpy side is generated live. Entry point `run_fidelity_visual`; tests (R-free, synthetic stand-in fixture) in `tests/test_fidelity_visual.py`; CLI via `scripts/fidelity_visual.py`. See `simulations/FIDELITY.md`.
  - `semisynthetic.py` — Generates `SemiSyntheticTrajectoryDataset` from the numpy generator. Trajectory modes are **feature-set surgery** on per-stage differential indicators: group A is a random baseline trajectory; group B is a deterministic transform — `magnitude` scales δ (size), `orientation` is one global per-omic feature permutation (rotation), `shape` permutes interior stages only (bend), `translation` is a constant observed-space location offset, `none` is identical. `group_effect_size` is the unified knob (0 = null for every mode). Truth records per-stage/group indicators and per-omic δ.
  - `evaluation.py` — Runs one dataset through the full MOTCO pipeline (integration → design → `estimate_difference` → optional RRPP). Returns `SimulationEvaluationResult`.
  - `grid.py` — Orchestrates replicate runs across a `SimulationGrid` of `SimulationCell` parameter combinations (`generator_params` + `evaluation_params`). Supports resumable JSONL persistence, Type I / power grid enumeration, and rejection-rate summarization.
  - `specificity.py` — Dominant-specificity instrumentation: per mode, measures the group-vs-stage projection and per-statistic RRPP rejection rates to confirm each mode predominantly moves its target statistic (`magnitude`→`delta`, `orientation`→`angle`, `shape`→`shape`).
  - `showcase.py` — Illustrative demo (not the power study): generates one dataset per `trajectory_mode` from a shared baseline (same seed) and renders a per-scenario 2-component PLS-DA trajectory figure. `run_trajectory_showcase()` is the entry point; driven by `scripts/trajectory_showcase.py`.
  - `study/` — Declarative, cluster-executable trajectory power study (the engine behind the Type I / power validation of the `delta`/`angle`/`shape` tests). Submodules: `config` (`StudyConfig` + acceptance targets), `enumerate`, `sharding`, `merge`, `summary`, `report`, `targets`. See `simulations/study/README.md` — the operational handbook for running it locally or on SLURM.

The trajectory pipeline (group differences) is split across three modules — `design` → `trajectory` → `permutation`:

- **`stats/design.py`** — Design construction.
  1. `get_model_matrix(X, group_col, level_col, full)` — builds an intercept + dummy-coded design matrix with optional group×level interactions. Category order is deterministic (sorted string representation).
  2. `build_ls_means(group_levels, level_levels, full)` — generates the LS-mean row vectors consistent with the coding from `get_model_matrix`. Row order is group-major, level-minor.
  - `center_matrix(...)` — column-centering helper used by the estimators.

- **`stats/trajectory.py`** — Estimation + geometric metrics.
  - `estimate_difference(Y, model_matrix, LS_means, contrast)` — fits betas via normal equations (Cholesky → direct solve → lstsq fallback), then computes per-group trajectory size (`_estimate_size`), orientation (`_estimate_orientation` via eigendecomposition), and shape (`_estimate_shape` via iterative Procrustes GPA). Returns symmetric matrices `(deltas, angles, shapes)`. `estimate_betas`, `get_observed_vectors`, and `pair_difference` expose the beta/LS-mean prediction step directly.

- **`stats/permutation.py`** — `RRPP(Y, model_full, model_reduced, LS_means, contrast, permutations, n_jobs)` — permutes residuals of the reduced model and calls `estimate_difference` for each permutation. Serial by default; parallel via `multiprocessing.Pool` using `_RRPPWorker` (a picklable callable).

### Key conventions

- All model matrices must include an intercept column. `get_model_matrix` always prepends one.
- `contrast` is a `list[list[int]]`: each inner list enumerates the LS-mean row indices belonging to one group (group-major, level-minor indexing matches `build_ls_means`).
- `group_col` and `level_col` must always be passed explicitly — no defaults anywhere.
- `MOTCO_TEST_PERMS` controls permutation count in regression tests (default `10000`). `MOTCO_N_JOBS` controls RRPP parallelism (default `1`; use `-1` for all CPUs).
- Regression test reference values are in `tests/data/results_example1.csv` and `tests/data/results_example2.csv`. `tests/data/reference/` contains the R script used to generate them, not CSV outputs.
- The package is built with `hatchling`; version is sourced from `src/motco/__init__.py`.
