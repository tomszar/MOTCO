## Context

`plsda_doubleCV` (`src/motco/stats/pls.py:26`) implements a double-nested cross-validation procedure intended to (a) select the optimal number of PLS latent variables via inner CV and (b) report an honest classification AUROC via outer CV. The canonical procedure is:

```
for outer fold k = 1..K:
    train_outer = all data except fold k
    for n_LV in candidates:
        mean_inner_auroc[n_LV] = mean AUROC over V inner CV folds of train_outer
    n_LV*_k = argmax over n_LV of mean_inner_auroc
    test_auroc_k = AUROC of PLS(n_LV*_k) fit on train_outer, scored on fold k

report mean ± std of {test_auroc_k} across k = 1..K
(repeat the K-fold partition n_repeats times for stability)
```

The current implementation deviates in two places, both biasing the reported AUROC upward.

**Deviation 1 — inner-loop model selection (BUG-1).** Inside each outer fold's inner CV (`pls.py:117-137`), the code records per inner fold the `(argmax n_LV, max AUROC)`. After the inner loop completes (`pls.py:139-140`), it picks the inner *fold* with the highest AUROC and uses that fold's `n_LV` as `n_components` for the outer test. This means: (i) AUROC is never averaged across inner folds, defeating the variance-reduction purpose of inner CV; (ii) the chosen `n_LV` reflects a single fortunate inner fold; (iii) when AUROC ties at saturation (e.g. 1.0 across multiple `n_LV` values within a fold), `list.index` returns the first occurrence, locking `n_LV = 2` deterministically.

**Deviation 2 — per-repeat outer aggregation (BUG-2).** After the K outer folds for one repeat have been evaluated (`pls.py:148-155`), the code picks the *single* outer fold with the highest test AUROC and writes only its `(n_LV, AUROC)` into `model_table`, discarding the other `K-1` test AUROCs. The retained model is the cherry-picked outer-fold model. The reported AUROC per repeat is therefore the *maximum* over K test scores, not the mean.

The exploration session (transcript) confirmed that, with the toy dataset, the dominant cause of "AUROC = 1 on every repeat" is BUG-2: even with substantially harder data, at least one outer fold per repeat lands at 1.0 with high probability, and max-of-K reporting pins the value.

## Goals / Non-Goals

**Goals:**
- Make the inner-loop selection use mean AUROC across V inner folds.
- Make the per-repeat reported AUROC be the mean across the K outer folds, with sample std reported alongside.
- Make the `n_LV` recorded per repeat be the mode of the K outer folds' selected `n_LV*` (with deterministic tie-break).
- Make `models["models"][i]` a model refit on the full input `(X, y)` using rep `i`'s chosen `n_LV`, so that VIP scores derived from it are computed on a model trained on all available data, not on a CV fold.
- Preserve the public signature and the table's row count (one row per repeat).
- Preserve positional access semantics: column at `iloc[:, 1]` is still `LV`, column at `iloc[:, 2]` is still `AUROC` (now a mean).

**Non-Goals:**
- Changing the function signature, defaults, or call conventions.
- Reporting at a finer grain than per-repeat (no per-fold rows; that would break callers asserting `table.shape[0] == n_repeats`).
- Adding a "final consensus" model across repeats.
- Changing `RepeatedStratifiedKFold` for any other CV strategy.
- Tuning the toy dataset (covered separately by `2026-05-08-retune-toy-dataset-difficulty`).

## Decisions

### D1 — Inner-loop selection: mean across folds, then argmax

For each candidate `n_LV ∈ {1, …, max_components - 1}`, compute the inner-CV mean AUROC across the V inner folds; pick the `n_LV` with the highest mean. Tie-break = smaller `n_LV` (parsimony; matches the deterministic behaviour of `numpy.argmax` and `list.index`).

Alternative considered: median or 1-SE rule. Rejected because (a) mean is the standard nested-CV reference, (b) the 1-SE rule biases toward smaller `n_LV` in a way that requires its own design discussion, and (c) the existing CLI/notebook narrative is built on a simple "best AUROC" framing.

### D2 — Per-repeat aggregation: mean and sample std across outer folds

The reported `AUROC` per repeat is the arithmetic mean of the K outer-fold test AUROCs; the new `AUROC_std` column reports their sample standard deviation (`numpy.std(..., ddof=1)`). With K = 5 (default), `ddof=1` is preferred; if K < 2, `AUROC_std` is `numpy.nan`.

Alternative considered: standard error (`std / sqrt(K)`). Rejected because std is the more direct fold-to-fold dispersion measure and is the convention in nested-CV reports; SE is recoverable from the column.

Alternative considered: storing the full K-vector. Rejected because it complicates positional column access and the test/CLI/notebook contracts.

### D3 — Per-repeat LV choice: mode with parsimony tie-break

Each of the K outer folds runs its own inner CV and selects its own `n_LV*`. Per repeat, `LV` is the mode of these K values; if multiple values tie, the smaller is reported. The reported `LV` is therefore an integer, preserving the dtype expected by `cmd_plsr` and the notebook's `mode()[0]` aggregation.

Alternative considered: median. Rejected because median can return non-integer values when K is even, breaking integer-typed downstream consumers.

### D4 — Per-repeat model: refit on full data with chosen `n_LV`

`models["models"][i]` is a `PLSRegression(n_components=LV_i, scale=True, max_iter=1000)` fit on the full input `(X, y)`. Rationale:

- VIP scores via `calculate_vips` should reflect the full available signal, not a single fold's training subset.
- It mirrors the standard "select hyperparameters by CV; refit final model on all data" workflow.
- It aligns with `cmd_plsr --out-scores`, which already does an independent full-data refit via `fit_plsda_transform` using the modal `n_LV` from the table.

Alternative considered: store the K outer-fold models per repeat (`list[list[PLSRegression]]`). Rejected because it changes the type of `models["models"]` and breaks existing callers.

Alternative considered: store the outer-fold model whose `n_LV*` matches the modal `LV`. Rejected because it propagates a fold's training-set bias into VIPs and is harder to explain.

### D5 — Schema additions are positional and additive

Column order in the returned `table`: `["rep", "LV", "AUROC", "AUROC_std"]`. The first three preserve their positions, types, and meanings (with AUROC now interpreted as a mean across outer folds rather than a max). `AUROC_std` is appended; existing positional readers (`tests/test_pls.py`, `cmd_plsr`, the notebook) are unaffected.

### D6 — `random_state` semantics unchanged

`random_state` continues to be passed only to `RepeatedStratifiedKFold`. The full-data refit of `models["models"][i]` uses no randomness beyond the deterministic PLS fit on `(X, y)`. The function remains deterministic for fixed inputs, including across the change.

## Risks / Trade-offs

- **[Risk] Headline AUROC numbers in prior outputs become non-comparable.** Anyone who has saved `cv_result['table']` from a previous run is reading max-of-K values; new outputs will report mean-of-K. *Mitigation:* the proposal calls this out explicitly; the change should be merged before any results are quoted in writeups.
- **[Risk] Tutorial users see the toy data's AUROC drop visibly.** With the bundled toy dataset (effect size 2.0), the mean-of-K AUROC will be very close to 1.0 with very small std, but no longer pinned. The values are still uninformatively high until the toy is retuned. *Mitigation:* a separate change (`2026-05-08-retune-toy-dataset-difficulty`) is sequenced after this one to lower toy difficulty and reveal honest variance.
- **[Risk] VIP scores change.** `models["models"][i]` now reflects a full-data refit instead of a cherry-picked fold model, so `calculate_vips` produces different numbers. This is a correctness improvement (less bias), not a regression. *Mitigation:* update tests that assert specific VIP values; document in the changelog.
- **[Trade-off] Modal `LV` discards information.** Mode collapses K possibly distinct fold choices into one integer. The full distribution is recoverable in principle (run the function again with `progress=True` and inspect logs) but is not retained in the table. Acceptable because the table is a summary, not a per-fold ledger.
- **[Trade-off] Refitting per repeat costs one extra PLS fit per repeat.** With `n_repeats = 30`, that's 30 extra fits on the full dataset. Cost is negligible relative to the K × n_repeats × max_components inner fits already performed.
- **[Trade-off] BUG-1 is selection bias on `n_LV`, not on the test AUROC.** Even before this fix, the outer test AUROCs were honest (the test data was held out from the `n_LV` selection). What was biased was the chosen `n_LV` and, via BUG-2, what was reported. Fixing BUG-1 alone would slightly stabilize and possibly *raise* outer test AUROCs (better-chosen `n_LV*`); fixing BUG-2 alone would lower the reported number. Bundling both lands the change in one coherent state.
