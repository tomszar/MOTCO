## Context

`plsda_doubleCV` in `stats/pls.py` runs a double cross-validation to evaluate model stability: the outer loop (CV2) selects the best number of latent variables per repeat; the inner loop (CV1) uses AUROC to rank LV counts. It returns a `models_table` dict with a `"table"` DataFrame (rep, LV, AUROC) and a `"models"` list of fitted `PLSRegression` objects — one per repeat, each fitted on a CV2 training split (~87.5% of data).

None of those models are fitted on the full dataset, so their `x_scores_` cannot be used directly as a latent representation of all samples. There is no existing path in the CLI or the Python API to produce a latent score matrix.

## Goals / Non-Goals

**Goals:**
- Expose a `fit_plsda_transform(X, y, n_components)` function in `pls.py` that fits on full data and returns scores.
- Add `--input` / `--metadata` / `--label-col` / `--out-scores` / `--n-components` to `motco plsr`.
- Preserve the existing `--data` / `--x` / `--y` single-matrix path without modification.

**Non-Goals:**
- Changing the double-CV logic or its defaults.
- Producing per-feature or per-component loading matrices (out of scope for this change).
- Adding a `--out-scores` path for the single-matrix mode that bypasses double-CV (scores are always derived from CV-selected LV count).

## Decisions

### D1 — LV count for final model: modal value from CV table

After double-CV produces a table of 30 best LV counts (one per repeat), we fit the final model using the **modal** LV count. Alternative considered: use the repeat with the highest AUROC. Modal is more robust to a single lucky split inflating the LV count; it represents the most frequently selected solution across diverse data splits.

When `--n-components` is provided, it overrides this entirely and the CV table is used only for the performance report.

### D2 — Multi-omics standardization: per-layer z-score, then concatenate

When `--input` receives multiple files, each layer is standardized independently (subtract column mean, divide by column std; std=0 columns set to 1) before horizontal concatenation. This mirrors the `_concat_integration` logic already used in `evaluation.py`, ensuring consistent behavior between the CLI and the Python evaluation harness.

Alternative considered: no standardization (user responsibility). Rejected because omics layers have vastly different scales; raw concatenation would give methylation (bounded [0,1]) negligible influence compared to expression.

### D3 — New function in `pls.py`, not inlined in CLI

`fit_plsda_transform` is added as a public function in `stats/pls.py` rather than inlined in `cli.py`. This makes it testable in isolation and available to Python API users who want to produce a latent space without going through the CLI.

### D4 — `--out-scores` works in both single-matrix and multi-omics modes

When `--data` / `--x` / `--y` is used and `--out-scores` is present, we apply the same final-fit logic (modal LV → `fit_plsda_transform` → save scores). This keeps the capability orthogonal to the input mode.

## Risks / Trade-offs

- **[Risk] Final model overfits relative to CV estimate** — The final model is fitted on 100% of data; the AUROC in the table came from held-out splits. Users must understand the score matrix reflects in-sample fit, not generalization performance. Mitigation: document clearly in CLI help text.
- **[Risk] Modal LV may differ substantially from the AUROC-optimal repeat** — In small datasets with high variance, the modal LV could be suboptimal. Mitigation: `--n-components` override lets the user impose a different choice.
- **[Trade-off] Double-CV still runs even when only `--out-scores` is wanted** — We always run double-CV to get the LV count before fitting the final model. This adds runtime. Alternative (skip CV, use user-specified `--n-components` only) would lose the performance table, which is useful for diagnosis.
