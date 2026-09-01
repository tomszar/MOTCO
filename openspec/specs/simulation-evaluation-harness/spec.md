# simulation-evaluation-harness Specification

## Purpose
Define the per-replicate evaluation layer that runs one semi-synthetic trajectory dataset through MOTCO integration, trajectory statistics, optional RRPP p-value estimation, and result metadata capture.
## Requirements
### Requirement: Harness evaluates one semi-synthetic trajectory dataset
MOTCO SHALL provide a simulation evaluation harness that accepts one `SemiSyntheticTrajectoryDataset` and returns a structured evaluation result.

#### Scenario: Successful single-dataset evaluation
- **WHEN** a caller provides a valid semi-synthetic trajectory dataset and evaluation parameters
- **THEN** the harness returns observed trajectory statistics and evaluation metadata

#### Scenario: Result includes generator truth
- **WHEN** evaluation succeeds
- **THEN** the result includes the generator truth metadata from the input dataset

### Requirement: Harness supports initial integration methods
The harness SHALL construct the molecular latent space — the measurement substrate in which trajectory geometry is estimated — via a selectable integration method, operating on M-value-converted methylation and raw expression/proteomics. The production latent-space methods are **SNF** (graph-spectral embedding) and **PLS** (the transform of the omic features into the subspace that maximizes covariance with the stage label). `concat` is retained as a **baseline/diagnostic** path (standardized feature concatenation), not a constructed latent space. The viz down-projection (`plot_trajectory_from_*`) is display-only and distinct from this measurement space.

#### Scenario: Concatenated baseline integration
- **WHEN** the caller selects `concat` integration
- **THEN** the harness converts methylation to M-values, standardises all layers, and concatenates them into the outcome matrix
- **AND** the result metadata identifies `concat` as a baseline rather than a constructed latent space

#### Scenario: SNF integration
- **WHEN** the caller selects `snf` integration
- **THEN** the harness converts methylation to M-values and creates the latent space from SNF fusion and spectral embedding

#### Scenario: PLS integration
- **WHEN** the caller selects `pls` integration
- **THEN** the harness converts methylation to M-values, standardises all layers, fits PLS-DA conditioned on the stage label, and returns the PLS X-score matrix as the latent space
- **AND** the number of latent variables is selected by the double nested cross-validation (modal LV across repeats, parsimony tie-break) to secure a stable, non-overfitted molecular space
- **AND** the result metadata records the selected number of latent variables and the cross-validation parameters

#### Scenario: PLS integration is infeasible
- **WHEN** the caller selects `pls` integration but the sample provides too few observations per stage for the cross-validation
- **THEN** the harness raises a clear validation error

#### Scenario: Unsupported integration method
- **WHEN** the caller selects an unsupported integration method
- **THEN** the harness raises a clear validation error

### Requirement: Harness builds MOTCO trajectory design objects
The harness SHALL construct model matrices, LS means, and trajectory contrasts from generated sample metadata.

#### Scenario: Design objects are derived from metadata
- **WHEN** the dataset metadata contains valid `group` and `stage` columns
- **THEN** the harness builds full and reduced model matrices, LS means, and a two-group trajectory contrast

#### Scenario: Missing metadata columns are rejected
- **WHEN** required metadata columns are missing
- **THEN** the harness raises a clear validation error

### Requirement: Harness estimates observed trajectory differences
The harness SHALL estimate observed `deltas`, `angles`, and `shapes` using MOTCO trajectory routines.

#### Scenario: Observed statistics are returned
- **WHEN** evaluation succeeds
- **THEN** the result includes observed `deltas`, `angles`, and `shapes`

#### Scenario: Pairwise group statistic is exposed
- **WHEN** evaluation succeeds for two groups
- **THEN** the result includes scalar statistics for the generated group comparison

### Requirement: Harness optionally runs RRPP p-value estimation
The harness SHALL optionally run RRPP and compute empirical p-values for observed trajectory statistics.

#### Scenario: RRPP disabled
- **WHEN** the caller sets permutation count to 0
- **THEN** the harness returns observed statistics without RRPP p-values

#### Scenario: RRPP enabled
- **WHEN** the caller sets permutation count greater than 0
- **THEN** the harness runs RRPP and returns empirical p-values for available statistics

#### Scenario: P-values use plus-one correction
- **WHEN** RRPP p-values are computed
- **THEN** the harness uses `(1 + count(null >= observed)) / (1 + n_permutations)`

### Requirement: Harness records evaluation metadata
The harness SHALL record parameters and runtime metadata needed for later grid aggregation.

#### Scenario: Evaluation metadata is returned
- **WHEN** evaluation succeeds
- **THEN** the result includes integration method, integration parameters, permutation count, runtime seconds, and evaluation parameters

### Requirement: Harness can compute orientation attribution from the evaluated PLS representation

The harness SHALL optionally compute orientation-attribution diagnostics for a PLS evaluation using the exact pooled standardized feature matrix and fitted PLS estimator that produced the trajectory score matrix. It MUST keep preprocessing, feature order, selected component count, and the PLS estimator fixed while calculating observed, PLS-captured, residual, and bootstrap values.

#### Scenario: Attribution is enabled for PLS

- **WHEN** a caller enables attribution diagnostics for a valid PLS evaluation
- **THEN** the harness returns attribution diagnostics calculated from the same standardized features and fitted PLS estimator used for trajectory measurement
- **AND** no second PLS fit or component selection is performed for attribution

#### Scenario: Attribution is disabled

- **WHEN** attribution diagnostics are not enabled
- **THEN** evaluation returns the existing statistics, p-values, geometry, and metadata without attribution work

#### Scenario: Attribution is requested for a non-PLS method

- **WHEN** a caller enables PLS orientation attribution with `concat` or `snf` integration
- **THEN** the harness rejects the incompatible configuration with a descriptive validation error

### Requirement: Attribution diagnostics are bounded and machine-readable

The harness SHALL return a JSON-safe diagnostic summary containing the effective attribution configuration, ordered transitions, observed/PLS-captured/residual transition metrics, configured top-k signed feature records in pooled standardized units and in original units wherever the fitted preprocessor supplies a positive per-feature scale, bootstrap sign and selection stability, and available recovery metrics against generator truth. It MUST NOT place a fitted estimator, full standardized matrix, full bootstrap matrix, or unrestricted feature table in the persisted diagnostic payload.

#### Scenario: Compact diagnostic is produced

- **WHEN** attribution completes for an eligible replicate
- **THEN** the result contains compact transition and top-k feature records with feature identifiers, component labels, signs, ranks, and stability values
- **AND** each feature record carries its standardized effect and, when a per-feature scale is available, its original-unit effect, with the unit basis labeled so M-value methylation is not read as beta values
- **AND** the payload records the attribution seed, bootstrap count, top-k value, selected PLS component count, feature order signature, and diagnostic schema version

#### Scenario: Generator truth is available

- **WHEN** the generated dataset identifies features whose group-stage differential pattern changed
- **THEN** the diagnostic reports top-k precision, recall, and selection counts against that truth using a documented truth definition that includes propagated omic effects

#### Scenario: A transition is unavailable

- **WHEN** either group has a zero or near-zero transition for an attribution component
- **THEN** the diagnostic marks the affected metrics unavailable and preserves transition identity and path lengths without treating unavailable values as zero

### Requirement: PLS fitting remains single-pass within an evaluation

The harness SHALL fit one final pooled PLS estimator after component selection and SHALL use that estimator's training scores for trajectory measurement and that same estimator for attribution projection and reconstruction.

#### Scenario: Scores and attribution share one final fit

- **WHEN** a PLS evaluation with attribution succeeds
- **THEN** the score matrix equals the training scores of the estimator supplied to attribution
- **AND** the selected component count and score values remain equivalent to a PLS evaluation with attribution disabled under the same inputs and seeds

### Requirement: Harness summarizes each permutation null

Whenever the harness runs RRPP, it SHALL return a compact per-statistic summary of the permutation null distribution alongside the observed statistics and p-values. The summary MUST locate the observed statistic against its own null: for each statistic with a null distribution it carries the number of retained permutation draws, the null mean, the null standard deviation, and null quantiles including the median and the alpha-level upper critical value. The summary MUST be JSON-safe scalars only, and non-finite draws MUST be excluded from the summary with the retained count reported.

The summary is separate from the existing opt-in retention of full null distributions: it is produced whenever permutations are run, whereas the full null vectors remain available only when the caller asks for them.

#### Scenario: Summary accompanies every RRPP evaluation

- **WHEN** a caller evaluates a dataset with permutation count greater than 0
- **THEN** the result carries a null summary for each statistic that has a null distribution
- **AND** each summary reports the retained draw count, mean, standard deviation, and quantiles for that statistic

#### Scenario: Summary is independent of full-distribution retention

- **WHEN** a caller runs RRPP without requesting retention of the full null distributions
- **THEN** the result still carries the compact null summary
- **AND** the full null vectors are absent

#### Scenario: No permutations means no summary

- **WHEN** a caller sets permutation count to 0
- **THEN** the result carries no null summary and the existing observed statistics are unchanged

#### Scenario: Summary does not alter the test

- **WHEN** the null summary is produced for an evaluation
- **THEN** the observed statistics, p-values, permutation draws, and every other field of the result are identical to those the same inputs produced before the summary existed
