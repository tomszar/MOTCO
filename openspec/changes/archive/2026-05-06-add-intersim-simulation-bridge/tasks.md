## 1. Package Structure

- [x] 1.1 Create `src/motco/simulations/` with an `__init__.py` that exposes the InterSIM bridge API
- [x] 1.2 Add an InterSIM bridge module under `src/motco/simulations/` for Python-side parameter handling, dependency checks, subprocess execution, and result normalization
- [x] 1.3 Add a package-owned R helper script under the `motco` package tree and ensure it is included in wheel/sdist builds

## 2. Python Data Contract

- [x] 2.1 Define an `InterSIMParams` model or equivalent function signature covering InterSIM-native parameters and `seed`
- [x] 2.2 Define an `InterSIMResult` dataclass containing methylation, expression, proteomics, sample IDs, clusters, and metadata
- [x] 2.3 Implement validation that matrix rows and cluster assignments have matching sample IDs and row counts
- [x] 2.4 Implement clear MOTCO-owned exceptions for missing dependencies, R process failure, invalid parameters, and malformed helper output

## 3. Rscript Bridge

- [x] 3.1 Implement an availability check for `Rscript`
- [x] 3.2 Implement an availability check for `requireNamespace("InterSIM", quietly = TRUE)`
- [x] 3.3 Implement the R helper script to call `set.seed(seed)` and `InterSIM::InterSIM(...)`
- [x] 3.4 Have the R helper write methylation, expression, protein, and cluster assignment CSV files to a Python-provided output directory
- [x] 3.5 Implement Python subprocess invocation with captured stdout/stderr and useful error messages
- [x] 3.6 Implement Python loading and normalization of the helper output into `InterSIMResult`

## 4. Tests

- [x] 4.1 Add pure Python unit tests for parameter translation and command construction without requiring R
- [x] 4.2 Add pure Python unit tests for output loading and sample-alignment validation using small fixture CSV files
- [x] 4.3 Add tests for missing `Rscript` and missing InterSIM package error handling using mocks
- [x] 4.4 Add an integration smoke test that skips when InterSIM is unavailable and proves Python can invoke InterSIM when available
- [x] 4.5 Add a reproducibility smoke test that runs the same seed twice and compares returned matrices and clusters when InterSIM is available

## 5. Documentation and Verification

- [x] 5.1 Document the InterSIM bridge API and required R dependency in docs or module docstrings
- [x] 5.2 Document how to install InterSIM for local development
- [x] 5.3 Run `uv run pytest tests/ -m "not slow" --tb=short`
- [x] 5.4 Run `uv run ruff check src/ tests/`
- [x] 5.5 Run `uv run mypy src/motco/`
- [x] 5.6 Run an InterSIM-enabled smoke test locally if R InterSIM is installed
