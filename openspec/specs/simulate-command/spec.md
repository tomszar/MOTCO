# simulate-command Specification

## Purpose
Define the `motco simulate` CLI contract for generating tutorial-ready multi-omics toy datasets and trajectory design files.

## Requirements

### Requirement: motco simulate subcommand
`motco simulate` SHALL be a CLI subcommand that generates a complete multi-omics toy dataset and trajectory design in a single invocation. It SHALL accept the following arguments:
- `--seed` (int, required): random seed passed to InterSIM and the semisynthetic generator.
- `--out-dir` (path, required): directory where all output files are written (created if absent).
- `--n-samples` (int, default 90): total sample count passed to InterSIM.
- `--trajectory-mode` (str, default `"orientation"`): one of `none`, `translation`, `magnitude`, `orientation`, `shape`.
- `--effect-size` (float, default 1.0): group effect size passed to `SemiSyntheticTrajectoryParams`.
- `--prop-affected-features` (float, default 0.1): proportion of features per omic layer that carry the injected group effect, passed to `SemiSyntheticTrajectoryParams.prop_affected_features`. Must lie in `[0, 1]`.

#### Scenario: Successful run produces all expected files
- **WHEN** `motco simulate --seed 42 --out-dir out/` is run with R and InterSIM available
- **THEN** `out/` contains: `methylation.csv`, `expression.csv`, `proteomics.csv`, `metadata.csv`, `model_full.csv`, `model_reduced.csv`, `ls_means.csv`, `contrast.json`, `truth.json`

#### Scenario: Output directory is created if absent
- **WHEN** `--out-dir` points to a non-existent path
- **THEN** the directory is created and files are written successfully

#### Scenario: Reproducibility via seed
- **WHEN** the command is run twice with the same `--seed` and parameters
- **THEN** all output CSV and JSON files are identical

#### Scenario: prop-affected-features flag is wired through to truth.json
- **WHEN** `motco simulate --prop-affected-features 0.1 ...` is run
- **THEN** `truth.json["affected_features"][layer]` for each layer has length within +/-1 of `round(n_features_in_layer * 0.1)`

#### Scenario: Invalid prop-affected-features value exits before invoking R
- **WHEN** `motco simulate --prop-affected-features 1.5 ...` is run
- **THEN** the command exits with a non-zero status, prints a message identifying the out-of-range value, and does NOT invoke R or InterSIM

### Requirement: Output file formats
The command SHALL write outputs in the following formats:

- `methylation.csv`, `expression.csv`, `proteomics.csv`: samples as rows (feature columns only, no sample_id; row order matches `metadata.csv`).
- `metadata.csv`: columns `sample_id`, `group`, `stage`, `cluster` (one row per sample, same order as omics files).
- `model_full.csv`, `model_reduced.csv`, `ls_means.csv`: numeric matrices written without index, column names omitted (pure numeric CSV), compatible with `motco de --model-full` / `--model-reduced` / `--ls-means`.
- `contrast.json`: JSON array of arrays (list of index lists), compatible with `motco de --contrast`.
- `truth.json`: JSON object recording the exact parameters used and the injected effect vectors.

#### Scenario: Omics files are sample-aligned
- **WHEN** the command completes successfully
- **THEN** `methylation.csv`, `expression.csv`, `proteomics.csv`, and `metadata.csv` all have the same number of rows in the same sample order

#### Scenario: Design files are compatible with motco de
- **WHEN** `motco de --Y latent.csv --model-full out/model_full.csv --model-reduced out/model_reduced.csv --ls-means out/ls_means.csv --contrast out/contrast.json` is run after simulate
- **THEN** `motco de` runs without error

### Requirement: Graceful failure when InterSIM is unavailable
When R is not on PATH or the InterSIM R package is not installed, `motco simulate` SHALL exit with a non-zero status and print a human-readable message identifying the missing dependency and how to install it.

#### Scenario: Rscript not found
- **WHEN** `Rscript` is not on PATH
- **THEN** the command exits with code 1 and a message indicating that Rscript is required

#### Scenario: InterSIM R package not installed
- **WHEN** Rscript is available but `requireNamespace("InterSIM")` returns FALSE
- **THEN** the command exits with code 1 and a message indicating the R package to install
