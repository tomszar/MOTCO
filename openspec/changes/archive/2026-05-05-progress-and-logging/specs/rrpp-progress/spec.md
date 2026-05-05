## ADDED Requirements

### Requirement: RRPP serial path displays a tqdm progress bar
`RRPP` SHALL accept a `progress: bool = True` keyword argument. When `progress=True` and the serial path is used (`n_jobs` is `None` or `1`), a tqdm progress bar SHALL be displayed on stderr showing completed permutations, the unit label "perm", and an estimated time remaining.

#### Scenario: Serial RRPP with progress enabled (default)
- **WHEN** a user calls `RRPP(..., permutations=999)` without passing `progress`
- **THEN** a tqdm bar appears on stderr ticking from 0 to 999 as permutations complete

#### Scenario: Serial RRPP with progress disabled
- **WHEN** a user calls `RRPP(..., progress=False)`
- **THEN** no progress bar or tqdm output appears on stderr

#### Scenario: RRPP in a non-TTY environment
- **WHEN** `RRPP` is called with `progress=True` but stderr is not a TTY (e.g., piped output)
- **THEN** tqdm auto-disables and no bar output is written

### Requirement: RRPP parallel path emits an INFO log instead of a bar
When `RRPP` runs the parallel path (`n_jobs != 1`), it SHALL emit a single `logging.INFO` message indicating the number of permutations and workers, and SHALL NOT display a tqdm bar.

#### Scenario: Parallel RRPP announces worker count
- **WHEN** a user calls `RRPP(..., n_jobs=-1, permutations=999)`
- **THEN** a single INFO log line is emitted (e.g., "Running 999 permutations across 8 workers") and no progress bar is shown

### Requirement: plsda_doubleCV outer loop displays a tqdm progress bar
`plsda_doubleCV` SHALL accept a `progress: bool = True` keyword argument. When `progress=True`, a tqdm progress bar SHALL be displayed on stderr tracking outer CV2 folds, with total set to `cv2_splits * n_repeats` and unit label "fold".

#### Scenario: PLS-DA CV with progress enabled (default)
- **WHEN** a user calls `plsda_doubleCV(..., cv2_splits=8, n_repeats=30)` without passing `progress`
- **THEN** a tqdm bar appears ticking from 0 to 240 as outer folds complete

#### Scenario: PLS-DA CV with progress disabled
- **WHEN** a user calls `plsda_doubleCV(..., progress=False)`
- **THEN** no progress bar or tqdm output appears on stderr

### Requirement: estimate_betas emits WARNING logs on numerical fallback
`estimate_betas` SHALL emit a `logging.WARNING` message whenever it falls back from the Cholesky solve to the direct solve, and a second `logging.WARNING` whenever it falls back from the direct solve to `lstsq`. These warnings SHALL be emitted regardless of the `progress` parameter or any CLI flag.

#### Scenario: Cholesky fallback is logged
- **WHEN** `estimate_betas` is called with a model matrix whose XtX is not positive definite (Cholesky fails)
- **THEN** a WARNING log is emitted before attempting the direct solve

#### Scenario: lstsq fallback is logged
- **WHEN** both Cholesky and direct solve fail (e.g., rank-deficient matrix)
- **THEN** a WARNING log is emitted before falling back to `lstsq`

### Requirement: CLI --verbose flag enables DEBUG logging
The `motco` CLI top-level parser SHALL accept a `--verbose` flag. When passed, `logging.basicConfig` SHALL be called with `level=logging.DEBUG` and `stream=sys.stderr` before dispatching to any subcommand. Without `--verbose`, no `basicConfig` call SHALL be made, leaving the root logger at the Python default (WARNING).

#### Scenario: --verbose routes log output to stderr
- **WHEN** a user runs `motco --verbose de ...`
- **THEN** DEBUG and INFO log messages from all `motco.*` loggers appear on stderr

#### Scenario: Without --verbose, INFO logs are suppressed
- **WHEN** a user runs `motco de ...` without `--verbose`
- **THEN** no INFO or DEBUG log messages appear (WARNING messages from fallbacks still appear if a handler is configured by the caller)
