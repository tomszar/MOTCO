## MODIFIED Requirements

### Requirement: motco simulate subcommand
`motco simulate` SHALL be a CLI subcommand that generates a complete multi-omics toy dataset and trajectory design in a single invocation, using the numpy generator (no R at runtime). It SHALL accept at least the following arguments:
- `--seed` (int, required): random seed for generation and group assignment.
- `--out-dir` (path, required): directory where all output files are written (created if absent).
- `--n-samples` (int): total sample count.
- `--trajectory-mode` (str, default `"orientation"`): one of `none`, `translation`, `magnitude`, `orientation`, `shape`, with the feature-surgery semantics defined by the semisynthetic generator.
- `--effect-size` (float, default 1.0): group effect size (the magnitude scale `λ` for `magnitude`, the offset/feature-budget for the other non-null modes).
- per-omic cluster mean-shift controls (`δ`) for methylation, expression, and proteomics, with a convenience scalar that fans out to any not individually specified.

Effect-size and mean-shift values SHALL be validated (`>= 0`) and out-of-range values SHALL exit non-zero before any generation work.

#### Scenario: Successful run produces all expected files
- **WHEN** `motco simulate --seed 42 --out-dir out/` is run
- **THEN** `out/` contains: `methylation.csv`, `expression.csv`, `proteomics.csv`, `metadata.csv`, `model_full.csv`, `model_reduced.csv`, `ls_means.csv`, `contrast.json`, `truth.json`

#### Scenario: Output directory is created if absent
- **WHEN** `--out-dir` points to a non-existent path
- **THEN** the directory is created and files are written successfully

#### Scenario: Reproducibility via seed
- **WHEN** the command is run twice with the same `--seed` and parameters
- **THEN** all output CSV and JSON files are identical

#### Scenario: Invalid effect/shift value exits before generation
- **WHEN** a negative effect size or mean-shift is provided
- **THEN** the command exits with a non-zero status and a message identifying the out-of-range value, before doing generation work

## REMOVED Requirements

### Requirement: Graceful failure when InterSIM is unavailable
**Reason**: `motco simulate` no longer invokes R/InterSIM; the unavailable-dependency path is replaced by the missing-reference-cache handling in the new no-R requirement below.

## ADDED Requirements

### Requirement: simulate runs without an R dependency
Because generation uses the numpy generator and cached reference data, `motco simulate` SHALL run without `Rscript` or the R `InterSIM` package on the host.

#### Scenario: Runs with no Rscript on PATH
- **WHEN** `motco simulate` is run on a host with no `Rscript`
- **THEN** it generates outputs successfully from the cached reference data

#### Scenario: Missing reference cache is reported clearly
- **WHEN** the cached reference data artifact is absent
- **THEN** the command exits with a non-zero status and a message naming the missing artifact and how to regenerate it
