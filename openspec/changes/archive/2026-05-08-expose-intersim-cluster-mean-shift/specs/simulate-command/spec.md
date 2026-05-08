## MODIFIED Requirements

### Requirement: motco simulate subcommand
`motco simulate` SHALL be a CLI subcommand that generates a complete multi-omics toy dataset and trajectory design in a single invocation. It SHALL accept the following arguments:
- `--seed` (int, required): random seed passed to InterSIM and the semisynthetic generator.
- `--out-dir` (path, required): directory where all output files are written (created if absent).
- `--n-samples` (int, default 90): total sample count passed to InterSIM.
- `--trajectory-mode` (str, default `"orientation"`): one of `none`, `translation`, `magnitude`, `orientation`, `shape`.
- `--effect-size` (float, default 1.0): group effect size passed to `SemiSyntheticTrajectoryParams`.
- `--prop-affected-features` (float, default 0.1): proportion of features per omic layer that carry the injected group effect, passed to `SemiSyntheticTrajectoryParams.prop_affected_features`. Must lie in `[0, 1]`.
- `--delta-methyl` (float, default `None`): InterSIM `delta.methyl` (per-feature mean shift between clusters for methylation). When `None`, InterSIM's own default applies. *(New flag.)*
- `--delta-expr` (float, default `None`): InterSIM `delta.expr` for expression. When `None`, InterSIM's own default applies. *(New flag.)*
- `--delta-protein` (float, default `None`): InterSIM `delta.protein` for proteomics. When `None`, InterSIM's own default applies. *(New flag.)*
- `--cluster-mean-shift` (float, default `None`): convenience scalar that fans out to any of `--delta-methyl`/`--delta-expr`/`--delta-protein` that were not individually specified. *(New flag.)*

Each provided delta value SHALL be `>= 0`; out-of-range values exit before invoking R.

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
- **WHEN** `motco simulate --prop-affected-features 0.1 …` is run
- **THEN** `truth.json["affected_features"][layer]` for each layer has length within ±1 of `round(n_features_in_layer * 0.1)`

#### Scenario: Invalid prop-affected-features value exits before invoking R
- **WHEN** `motco simulate --prop-affected-features 1.5 …` is run
- **THEN** the command exits with a non-zero status, prints a message identifying the out-of-range value, and does NOT invoke R or InterSIM

#### Scenario: cluster-mean-shift fans out to all three per-omic deltas
- **WHEN** `motco simulate --cluster-mean-shift 0.7 …` is invoked with no per-omic delta flags
- **THEN** the resulting `InterSIMParams` has `delta_methyl == 0.7`, `delta_expr == 0.7`, and `delta_protein == 0.7`

#### Scenario: per-omic deltas override the cluster-mean-shift fanout
- **WHEN** `motco simulate --cluster-mean-shift 0.7 --delta-expr 1.2 …` is invoked
- **THEN** the resulting `InterSIMParams` has `delta_methyl == 0.7`, `delta_expr == 1.2`, `delta_protein == 0.7`

#### Scenario: no delta flags preserves InterSIM defaults
- **WHEN** `motco simulate --seed 0 --out-dir out/` is invoked with none of `--delta-*` or `--cluster-mean-shift`
- **THEN** the resulting `InterSIMParams` has `delta_methyl is None`, `delta_expr is None`, `delta_protein is None`

#### Scenario: Negative delta exits before invoking R
- **WHEN** `motco simulate --delta-methyl -0.1 …` is invoked
- **THEN** the command exits with a non-zero status, the message identifies the out-of-range value, and `check_intersim_available` is not called
