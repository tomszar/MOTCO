## 1. CLI — motco simulate subcommand

- [x] 1.1 Add `cmd_simulate(args)` to `src/motco/cli.py`: call `check_intersim_available()` first; on failure, raise `SystemExit` with message naming the dependency and install instructions
- [x] 1.2 Add `simulate` subparser to `build_parser`: `--seed` (int, required), `--out-dir` (str, required), `--n-samples` (int, default 90), `--trajectory-mode` (str, default `"orientation"`), `--effect-size` (float, default 2.0)
- [x] 1.3 Implement output in `cmd_simulate`: call `generate_semisynthetic_trajectory_from_intersim` then `build_simulation_trajectory_design`; write all 9 files to `--out-dir` (create dir if absent)
- [x] 1.4 Write omics CSVs with `sample_id` as first column (index); write `metadata.csv` with columns `sample_id, group, stage, cluster`; write matrix CSVs (model_full, model_reduced, ls_means) as headerless numeric CSVs; write `contrast.json` and `truth.json` as JSON

## 2. Pre-generated toy dataset

- [x] 2.1 Run `motco simulate --seed 42 --n-samples 90 --trajectory-mode orientation --effect-size 2.0 --out-dir examples/data/toy/` and verify all 9 files are created
- [x] 2.2 Verify toy dataset end-to-end: run SNF on the three omics files → get `latent_snf.csv`; run `motco de` with the design files → confirm no errors and angle between groups is > 0
- [x] 2.3 Commit `examples/data/toy/` to the repository

## 3. Tests — motco simulate

- [x] 3.1 CLI integration test: `motco simulate` with valid args produces all 9 expected output files
- [x] 3.2 CLI test: same seed produces identical output on two runs (reproducibility)
- [x] 3.3 CLI test: missing R / InterSIM produces `SystemExit` with a message containing "InterSIM" (mock `check_intersim_available` to return unavailable)
- [x] 3.4 CLI test: `motco de` runs without error using the model_full, model_reduced, ls_means, contrast files produced by simulate (integration smoke test)

## 4. Notebook — rewrite motco_example.ipynb

- [x] 4.1 Replace `examples/motco_example.ipynb` with a new notebook containing 6 sections as specified: data generation, PLS-DA latent space, SNF latent space, trajectory design, differential trajectory analysis, visualization
- [x] 4.2 Section 1 (PLS): load toy data, concatenate + standardize omics, run `plsda_doubleCV(y=stage)`, fit final model, produce score matrix Y_pls
- [x] 4.3 Section 2 (SNF): run `get_affinity_matrix` + `SNF` + `get_spectral` on toy omics to produce Y_snf
- [x] 4.4 Section 3 (Design): load pre-generated `model_full.csv`, `model_reduced.csv`, `ls_means.csv`, `contrast.json` directly from `examples/data/toy/`
- [x] 4.5 Section 4 (DE): run `estimate_difference` and `RRPP` on both Y_pls and Y_snf; print angle, delta, shape statistics and p-values
- [x] 4.6 Section 5 (Viz): call `plot_trajectory_from_data` (or `plot_trajectories`) on both latent spaces to render group trajectory plots
- [x] 4.7 Execute notebook end-to-end and confirm no errors; clear outputs before committing

## 5. README — quick-start section

- [x] 5.1 Add "Quick start" section near the top of `README.md` with the 4-command CLI pipeline using `examples/data/toy/` files
- [x] 5.2 Include note that `motco simulate` requires R + InterSIM with the install command: `install.packages("InterSIM", repos = c("https://cran.r-universe.dev", "https://cloud.r-project.org"))`

## 6. Pre-commit gate

- [x] 6.1 Run `uv run ruff check src/ tests/` — no errors
- [x] 6.2 Run `uv run mypy src/motco/` — no new type errors
- [x] 6.3 Run `MOTCO_TEST_PERMS=99 uv run pytest tests/ -m "not slow" --tb=short` — all pass
