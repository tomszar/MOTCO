## Why

`motco plsr` currently only outputs a cross-validation performance table and VIP scores — it cannot produce a latent space (score matrix) for downstream analysis. To use PLS-DA as the supervised dimension-reduction step feeding into `motco de`, we need it to fit a final model on the full dataset and export the X scores as a usable outcome matrix Y.

## What Changes

- **`stats/pls.py`** — add `fit_plsda_transform(X, y, n_components) -> np.ndarray`: fits a `PLSRegression` on full X and returns `x_scores_` (n_samples × n_components).
- **`cli.py` — `plsr` subcommand** — add:
  - `--input` (repeatable): multi-omics mode; each file is standardized (z-score per feature) and concatenated horizontally before PLS.
  - `--metadata` + `--label-col`: provide y labels from a metadata CSV when using `--input`.
  - `--out-scores`: path to save the final-fit latent space (n_samples × n_components CSV).
  - `--n-components` (optional): override the modal LV count selected from double-CV.
- The existing `--data` / `--x` / `--y` single-matrix path is **unchanged**.

## Capabilities

### New Capabilities

- `plsr-latent-space`: Supervised multi-omics dimensionality reduction via PLS-DA. Accepts multiple omics input files, standardizes and concatenates them, runs double cross-validation to select the optimal number of latent variables, then fits a final model on all data and exports the X score matrix as a CSV for use as Y in `motco de`.

### Modified Capabilities

- `plsr-double-cv`: The CLI argument surface for `motco plsr` gains four new arguments (`--input`, `--metadata`, `--label-col`, `--out-scores`, `--n-components`). Existing arguments and their behavior are unchanged; this is additive only.

## Impact

- `src/motco/stats/pls.py`: new public function `fit_plsda_transform`.
- `src/motco/cli.py`: extended `cmd_plsr` and `build_parser` for the `plsr` subcommand.
- `tests/`: new tests for `fit_plsda_transform` and the new CLI paths.
- No breaking changes; no new package dependencies.
