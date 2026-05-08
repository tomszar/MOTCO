## 1. Inner-loop selection (BUG-1)

- [x] 1.1 In `plsda_doubleCV` (`src/motco/stats/pls.py`), replace the per-inner-fold "argmax + max" recording with a `(V × n_LV_candidates)` matrix of inner-fold AUROCs
- [x] 1.2 After the inner loop, compute `mean_per_n_LV = aurocs.mean(axis=0)` and set `n_LV*_k = int(np.argmax(mean_per_n_LV)) + 1` (parsimony tie-break: `argmax` returns the first occurrence)
- [x] 1.3 Remove the `cv1_table` two-column scratchpad; replace with the `(V × N_LV)` array described above
- [x] 1.4 Confirm that with `cv1_splits=2, max_components=3` the function still runs (covered by `tests/test_validation.py`)

## 2. Outer-loop aggregation (BUG-2)

- [x] 2.1 Replace the `cv2_table` two-column scratchpad with two same-length lists per repeat: `outer_aurocs: list[float]` and `outer_n_lv: list[int]`
- [x] 2.2 At the end of each repeat, compute `mean_auroc = float(np.mean(outer_aurocs))`, `std_auroc = float(np.std(outer_aurocs, ddof=1)) if len(outer_aurocs) > 1 else float("nan")`, and `mode_lv = _modal_int_with_parsimony(outer_n_lv)`
- [x] 2.3 Add a small private helper `_modal_int_with_parsimony(values: list[int]) -> int` that returns the most frequent value, breaking ties by smaller value
- [x] 2.4 Remove the cherry-pick logic at `pls.py:148-155`; do not store any per-fold model in the per-repeat output

## 3. Per-repeat model refit (D4)

- [x] 3.1 After computing `mode_lv` per repeat, fit `PLSRegression(n_components=mode_lv, scale=True, max_iter=1000).fit(X_arr, yd_arr)` on the full input where `yd_arr` is the one-hot encoded `y`
- [x] 3.2 Append the refit model to `best_models`; ensure exactly `n_repeats` models are returned in `models["models"]`
- [x] 3.3 Confirm `calculate_vips(models["models"][i])` runs without error for `i in range(n_repeats)` (the existing test `tests/test_pls.py:32` covers this; verify it still passes)

## 4. Schema and return shape

- [x] 4.1 Initialize `model_table` with columns `["rep", "LV", "AUROC", "AUROC_std"]` (was: `["rep", "LV", "AUROC"]`)
- [x] 4.2 Populate columns by name where possible; preserve positional access at indices 0–2 for backward compatibility
- [x] 4.3 Verify `table.shape == (n_repeats, 4)` and dtypes: `rep` and `LV` are integer, `AUROC` and `AUROC_std` are float
- [x] 4.4 Update the docstring's "Returns" section to describe the four columns and the per-repeat aggregation semantics

## 5. Tests — `tests/test_pls.py`

- [x] 5.1 Update the schema assertion `result["table"].shape[0] == n_repeats` to also check `result["table"].shape[1] == 4`
- [x] 5.2 Add an assertion that `"AUROC_std"` column exists and all values are `>= 0` (or NaN)
- [x] 5.3 Add a new test `test_plsda_doubleCV_mean_aggregation_is_honest`: synthetic `(X, y)` with moderate SNR (e.g. two Gaussians shifted by 0.5 σ in 20 features) where ground-truth Bayes-optimal AUROC is ~0.85; assert `result["table"]["AUROC"].mean() < 0.99` (would fail under max-of-K aggregation) and `result["table"]["AUROC_std"].mean() > 0.0`
- [x] 5.4 Add a test `test_plsda_doubleCV_models_refit_on_full_data`: assert that for each `i`, `result["models"][i].x_scores_.shape[0] == X.shape[0]` (full-data refit, not a fold model)
- [x] 5.5 Verify existing tests at `tests/test_pls.py:33` (AUROC bounded in `[0, 1]`) and `tests/test_pls.py:47` (LV positive) still pass

## 6. Notebook narrative refresh

- [x] 6.1 In `examples/motco_example.ipynb`, update the markdown around the `cv_result['table']` print to reflect that AUROC is mean across K outer folds and `AUROC_std` shows fold-to-fold dispersion (no code change required)
- [x] 6.2 Re-execute the notebook end-to-end and clear outputs before committing

## 7. Pre-commit gate

- [x] 7.1 Run `uv run ruff check src/ tests/` — no errors
- [x] 7.2 Run `uv run mypy src/motco/` — no new type errors
- [x] 7.3 Run `MOTCO_TEST_PERMS=99 uv run pytest tests/ -m "not slow" --tb=short` — all pass
