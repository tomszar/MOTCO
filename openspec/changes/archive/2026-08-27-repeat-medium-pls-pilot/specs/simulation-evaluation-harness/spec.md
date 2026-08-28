## ADDED Requirements

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

