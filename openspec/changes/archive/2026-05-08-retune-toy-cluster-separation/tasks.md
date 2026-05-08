## 1. Regenerate Toy Dataset

- [x] 1.1 Confirm the working tree state and avoid staging unrelated local artifacts such as `.idea/`, `results/`, or experimental scripts.
- [x] 1.2 Regenerate `examples/data/toy/` with `motco simulate --seed 42 --n-samples 90 --trajectory-mode orientation --effect-size 1.0 --prop-affected-features 0.1 --cluster-mean-shift 0.10 --out-dir examples/data/toy/`.
- [x] 1.3 Verify `examples/data/toy/truth.json` records `group_effect_size == 1.0`, affected-feature list lengths near 10% per layer, and InterSIM delta metadata near `0.10` for methylation, expression, and proteomics.
- [x] 1.4 Confirm the required toy files are present and sample-aligned: `methylation.csv`, `expression.csv`, `proteomics.csv`, `metadata.csv`, `model_full.csv`, `model_reduced.csv`, `ls_means.csv`, `contrast.json`, `truth.json`.

## 2. Validate Tutorial Behavior

- [x] 2.1 Run `motco plsr` on the regenerated toy data with `y = stage` using moderate tutorial CV settings (for example `cv1_splits=3`, `cv2_splits=3`, `n_repeats=3`, `max_components=5`) and record the resulting AUROC table.
- [x] 2.2 Confirm stage classification is non-saturated: mean AUROC is less than `0.99` and at least one repeat has `AUROC_std > 0.0`.
- [x] 2.3 Generate a PLS-DA latent score file from the regenerated toy data for downstream `motco de` validation.
- [x] 2.4 Run `motco de` with at least 199 RRPP permutations on the regenerated latent scores and existing design files; confirm `0 < p_angle <= 0.1` and the latent-space angle remains at least 30° but below saturation.

## 3. Documentation and Specs

- [x] 3.1 Update `README.md` quick-start and regeneration command to include `--cluster-mean-shift 0.10` and example-friendly PLS-DA CV settings if needed.
- [x] 3.2 Update `examples/motco_example.ipynb` markdown to show the new regeneration command and explain that stage classification is intentionally moderate rather than perfect.
- [x] 3.3 If any observed validation bands differ from the proposal/spec assumptions, update this change's `design.md` and `specs/toy-dataset/spec.md` before implementation is considered complete.

## 4. Tests and Quality Gates

- [x] 4.1 Add or update tests that assert the bundled `truth.json` records the canonical toy generation parameters, including cluster mean shift metadata.
- [x] 4.2 Add or update tests that check the bundled toy files remain present, parseable, and row-aligned.
- [x] 4.3 Run `uv run ruff check src/ tests/`.
- [x] 4.4 Run `uv run mypy src/motco/`.
- [x] 4.5 Run `MOTCO_TEST_PERMS=99 uv run pytest tests/ -m "not slow" --tb=short`.
