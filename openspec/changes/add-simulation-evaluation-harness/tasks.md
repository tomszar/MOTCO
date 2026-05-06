## 1. Data Models and Public API

- [ ] 1.1 Add a simulation evaluation module under `src/motco/simulations/`
- [ ] 1.2 Define `SimulationEvaluationParams` for integration method, integration parameters, RRPP permutations, jobs, seed/progress settings, and output options
- [ ] 1.3 Define `SimulationEvaluationResult` for observed statistics, scalar group-pair statistics, p-values, latent matrix metadata, truth metadata, and runtime metadata
- [ ] 1.4 Export the new models and evaluation functions from `motco.simulations`

## 2. Integration Methods

- [ ] 2.1 Implement `concat` integration with deterministic feature scaling
- [ ] 2.2 Implement `snf` integration using existing affinity, SNF, and spectral embedding helpers
- [ ] 2.3 Validate integration parameters against sample count and feature availability
- [ ] 2.4 Add tests for `concat`, `snf`, and unsupported integration method validation

## 3. Trajectory Design Construction

- [ ] 3.1 Validate required metadata columns and omics/metadata row alignment
- [ ] 3.2 Build full and reduced model matrices from group/stage metadata
- [ ] 3.3 Build LS means from sorted group/stage levels
- [ ] 3.4 Build the two-group trajectory contrast from LS-mean row order
- [ ] 3.5 Add tests for design object shapes and contrast construction

## 4. Observed Statistics and RRPP

- [ ] 4.1 Run `estimate_difference` for observed trajectory statistics
- [ ] 4.2 Extract scalar group-pair `delta`, `angle`, and `shape` statistics
- [ ] 4.3 Support observed-only evaluation when permutations are 0
- [ ] 4.4 Run RRPP when permutations are greater than 0
- [ ] 4.5 Compute empirical p-values with plus-one correction
- [ ] 4.6 Handle unavailable shape statistics consistently when fewer than three stages are present
- [ ] 4.7 Add tests with low permutation counts for p-value plumbing

## 5. Result Metadata and Documentation

- [ ] 5.1 Record generator truth, integration parameters, permutation settings, and runtime seconds in the result
- [ ] 5.2 Document supported evaluation flow and integration methods in simulation docs
- [ ] 5.3 Document result schema and p-value calculation
- [ ] 5.4 Run `uv run pytest tests/ -m "not slow" --tb=short`
- [ ] 5.5 Run `uv run ruff check src/ tests/`
- [ ] 5.6 Run `uv run mypy src/motco/`
