## 1. Data Models and Grid Enumeration

- [ ] 1.1 Define simulation cell and grid parameter models
- [ ] 1.2 Define replicate result and summary result models
- [ ] 1.3 Implement deterministic cell ID generation
- [ ] 1.4 Implement Type I grid enumeration from baseline parameters and axes
- [ ] 1.5 Implement power grid enumeration from trajectory modes, effect sizes, and axes
- [ ] 1.6 Add tests for stable enumeration and validation errors

## 2. Replicate Execution

- [ ] 2.1 Implement deterministic seed derivation per cell/replicate
- [ ] 2.2 Implement local replicate runner using an injectable evaluator
- [ ] 2.3 Implement default runner path using the simulation evaluation harness
- [ ] 2.4 Implement configurable error policy for failed replicates
- [ ] 2.5 Add tests with fake evaluator for successful and failed replicates

## 3. Result Persistence and Resume

- [ ] 3.1 Choose initial persistence format (CSV, JSONL, or parquet with explicit dependency decision)
- [ ] 3.2 Implement result writer for one row per cell/replicate
- [ ] 3.3 Implement result reader
- [ ] 3.4 Implement parameter signature/hash recording
- [ ] 3.5 Implement resume logic that skips matching completed replicates
- [ ] 3.6 Implement mismatch detection and overwrite behavior
- [ ] 3.7 Add tests for write/read/resume/mismatch behavior

## 4. Summary Metrics

- [ ] 4.1 Implement rejection indicator calculation from p-values and alpha
- [ ] 4.2 Implement per-cell rejection rate summaries
- [ ] 4.3 Implement Monte Carlo standard error calculation
- [ ] 4.4 Implement handling for missing/unavailable statistics
- [ ] 4.5 Add tests for Type I and power-style summary outputs

## 5. Documentation and Verification

- [ ] 5.1 Document grid schema and example small Type I/power configurations
- [ ] 5.2 Document result persistence format and resume semantics
- [ ] 5.3 Document rejection-rate summary metrics
- [ ] 5.4 Run `uv run pytest tests/ -m "not slow" --tb=short`
- [ ] 5.5 Run `uv run ruff check src/ tests/`
- [ ] 5.6 Run `uv run mypy src/motco/`
