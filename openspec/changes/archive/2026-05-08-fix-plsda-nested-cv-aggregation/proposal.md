## Why

`plsda_doubleCV` deviates from canonical nested cross-validation in two places, both biasing the reported AUROC upward:

1. **Inner-loop model selection picks the best `(fold, n_LV)` pair instead of the n_LV that maximizes mean AUROC across inner folds.** The averaging step that justifies inner CV never happens; the chosen `n_LV` reflects a single fortunate inner fold rather than a stable estimate.
2. **Per-repeat reporting takes the maximum AUROC across the K outer folds instead of the mean.** The other `K-1` outer-fold test scores are discarded. With even moderately separable data, at least one outer fold lands at AUROC = 1.0, so the table pins at 1.0 regardless of overall fit quality.

The compounded effect: on the bundled toy dataset, every repeat reports AUROC = 1.0 and LV = 2, irrespective of seed. This is not a property of the data alone — even with substantially harder data, max-fold reporting will continue to mask realistic variance. Any downstream consumer of the table (CLI, notebook, anyone calling `plsda_doubleCV`) is reading an upwardly biased point estimate. The single model stored per repeat in `models["models"]` is itself the cherry-picked fold's model, propagating selection bias into VIP scores via `calculate_vips`.

## What Changes

- **`stats/pls.py`** — fix `plsda_doubleCV` to follow canonical nested CV:
  - Inner loop: for each candidate `n_LV`, compute mean AUROC across the `V` inner folds; pick the `n_LV` that maximizes this mean.
  - Outer loop: keep all K outer-fold test AUROCs and the per-fold selected `n_LV*`.
  - Per-repeat reporting: AUROC = mean across the K outer folds; LV = mode of per-fold selected `n_LV*` (tie-break = smaller); add an `AUROC_std` column = sample standard deviation across the K outer folds.
  - `models["models"][i]`: a single PLSRegression refit on the full input `(X, y)` with rep `i`'s chosen `n_LV` (not a cherry-picked outer-fold model).
- **Test suite** — update `tests/test_pls.py` to assert the new schema (4 columns, `LV` is modal int, `AUROC_std` non-negative) and to add a regression case where ground-truth AUROC is < 1, exercising mean-aggregation honesty.
- **CLI / notebook** — no required changes: `cmd_plsr` reads positional columns `iloc[:, 1]` (LV) and `iloc[:, 2]` (AUROC) which keep their meanings; the new `AUROC_std` column at position 3 is purely additive. Modal-LV computation in `cmd_plsr --out-scores` continues to work unchanged.

## Capabilities

### New Capabilities

- `plsda-nested-cv`: contract on the model-selection and reporting semantics of `plsda_doubleCV` — describes what each row of the returned `table` represents, how the `n_LV` choice per outer fold is made, what is stored in `models["models"]`, and why per-repeat aggregation is mean-based.

### Modified Capabilities

(none — `plsr-latent-space` and `rrpp-progress` describe orthogonal aspects of `pls.py` and remain unchanged.)

## Impact

- `src/motco/stats/pls.py`: rewrite of the `plsda_doubleCV` body (no signature change).
- `tests/test_pls.py`: schema assertions updated; new test case for mean-aggregation honesty on a moderate-SNR dataset.
- `tests/test_validation.py`: unaffected (input validation paths unchanged).
- `examples/motco_example.ipynb`: narrative around the printed CV table will become more accurate; no code change required (positional reads still work).
- `src/motco/cli.py`: unaffected (positional column reads + per-repeat models list semantics preserved).
- No external dependencies added or removed.
- Behavior change: anyone currently treating the AUROC column as a canonical performance estimate has been reading an upwardly biased value. Headline numbers in any prior outputs based on this function should be re-derived after this change lands.
