## 1. CLI — expose `--prop-affected-features` and lower `--effect-size` default

- [x] 1.1 In `src/motco/cli.py`, add a new argument to the `simulate` subparser: `--prop-affected-features` (float, default `0.1`); add an early validator at the top of `cmd_simulate` that exits with a non-zero status (before `check_intersim_available`) if the value is outside `[0, 1]`
- [x] 1.2 Lower the default of `--effect-size` from `2.0` to `1.0`
- [x] 1.3 Pass `prop_affected_features=args.prop_affected_features` into the `SemiSyntheticTrajectoryParams` constructor in `cmd_simulate`
- [x] 1.4 Update the `simulate` subparser help text to describe the new flag

## 2. Acceptance verification on seed 42

- [x] 2.1 Run `motco simulate --seed 42 --n-samples 90 --trajectory-mode orientation --effect-size 1.0 --prop-affected-features 0.1 --out-dir /tmp/toy_check/` and confirm all 9 expected files are produced
- [x] 2.2 Run the toy through the full pipeline (concatenate + standardize three omics → `fit_plsda_transform(y=stage, n_components=2)` → `motco de` with RRPP at 199+ permutations); record `angle_AB`, `p_angle`, `delta_AB`, `p_delta`, `shape_AB`, `p_shape`
- [x] 2.3 Confirm `0 < p_angle ≤ 0.1` and `30° ≤ angle_AB ≤ 85°` (orientation detectable but not pinned). If outside this band, adjust `--effect-size` within `[0.75, 1.5]` and re-test. Record final chosen value in `truth.json` (which is automatic via `motco simulate`).
- [x] 2.4 Note that `y = stage` AUROC remains at ~1.0 with `AUROC_std ≈ 0`. This is expected (see design D6); do NOT treat it as a failure. The fix-aggregation change ensures the value is honestly reported (mean across K outer folds), not max-cherry-picked.

## 3. Regenerate `examples/data/toy/`

- [x] 3.1 Delete the contents of `examples/data/toy/` (back up first if useful for diffing): `methylation.csv`, `expression.csv`, `proteomics.csv`, `metadata.csv`, `model_full.csv`, `model_reduced.csv`, `ls_means.csv`, `contrast.json`, `truth.json`
- [x] 3.2 Regenerate with `motco simulate --seed 42 --n-samples 90 --trajectory-mode orientation --effect-size <FINAL> --prop-affected-features 0.1 --out-dir examples/data/toy/`
- [x] 3.3 Verify `truth.json` records the new `group_effect_size` and that the `affected_features` lists reflect `prop_affected_features = 0.1` (within ±1 of the expected length per layer)
- [x] 3.4 Commit the regenerated files

## 4. Notebook narrative refresh

- [x] 4.1 In `examples/motco_example.ipynb`, update the data-generation markdown cell to show the new `motco simulate` command including the `--prop-affected-features` flag and lowered `--effect-size`
- [x] 4.2 Update the markdown cell preceding `cv_result['table']` to explicitly note that AUROC ≈ 1.0 with negligible std is expected here — `y = stage` classification on InterSIM cluster data is intentionally easy and is NOT what the toy is meant to make difficult; trajectory analysis (next sections) is where the realistic uncertainty lives
- [x] 4.3 Update the section-5 (Differential trajectory analysis) markdown cell to set expectations for the new numerical outputs: angle ~70° (was ~90°), p_angle ~0.03–0.05 (was 0.005), still detectable but no longer pinned
- [x] 4.4 Correct the trailing summary cell: remove the "delta p-values are expected to be non-significant in orientation mode" claim, which is true in the original feature space but NOT in the supervised PLS-DA latent space (and was incorrect at the prior `effect_size = 2.0` defaults too)
- [x] 4.5 Re-execute the notebook end-to-end; clear outputs before committing

## 5. README quick-start refresh

- [x] 5.1 If `README.md` quotes specific AUROC numbers in the quick-start, update them (or remove them — `y = stage` AUROC is not informative for the toy demonstration)
- [x] 5.2 If `README.md` shows the `motco simulate` invocation, add `--prop-affected-features 0.1` to it (or note the new defaults)

## 6. Tests

- [x] 6.1 Add a CLI test in `tests/test_cli.py`: `motco simulate --prop-affected-features 0.1 …` runs and `truth.json["affected_features"]["methylation"]` has expected length within ±1 of `round(367 * 0.1)`
- [x] 6.2 Add a CLI test: `motco simulate --prop-affected-features 1.5 …` exits with non-zero status and a message naming the out-of-range value, without invoking R (verify by mocking `check_intersim_available` and asserting it was not called, or by structuring the test so R isn't required)
- [x] 6.3 Verify existing simulate tests (output files exist; reproducibility on identical seed; missing R / InterSIM error path) still pass with the new defaults

## 7. Pre-commit gate

- [x] 7.1 Run `uv run ruff check src/ tests/` — no errors
- [x] 7.2 Run `uv run mypy src/motco/` — no new type errors
- [x] 7.3 Run `MOTCO_TEST_PERMS=99 uv run pytest tests/ -m "not slow" --tb=short` — all pass
