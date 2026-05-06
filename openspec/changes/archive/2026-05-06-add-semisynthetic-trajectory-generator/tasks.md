## 1. Data Models and Public API

- [x] 1.1 Add a semi-synthetic trajectory generator module under `src/motco/simulations/`
- [x] 1.2 Define `SemiSyntheticTrajectoryParams` covering trajectory mode, group effect size, group ratio, affected-feature configuration, group labels, and seed
- [x] 1.3 Define `SemiSyntheticTrajectoryDataset` containing methylation, expression, proteomics, sample metadata, and truth metadata
- [x] 1.4 Export the new parameter/result models and generator functions from `motco.simulations`

## 2. Clusters-as-Stages and Group Assignment

- [x] 2.1 Implement deterministic mapping from sorted InterSIM cluster labels to integer stages
- [x] 2.2 Preserve original InterSIM cluster labels in sample metadata
- [x] 2.3 Implement within-stage group assignment using configured group ratio and seed
- [x] 2.4 Validate that every stage can contain both comparison groups and raise clear errors otherwise
- [x] 2.5 Record stage mapping, group labels, group ratio, seed, and clusters-as-stages assumption in truth metadata

## 3. Affected Feature Selection

- [x] 3.1 Support explicit affected-feature lists per omics layer
- [x] 3.2 Support proportion-based affected-feature selection per omics layer using the generator seed
- [x] 3.3 Validate affected feature names and affected-feature proportions
- [x] 3.4 Record affected feature names per omics layer in truth metadata

## 4. Trajectory Effect Injection

- [x] 4.1 Implement `none` mode and zero-effect handling that preserves original InterSIM omics values
- [x] 4.2 Implement `translation` mode with stage-invariant group-specific shifts
- [x] 4.3 Implement `magnitude` mode with stage-proportional group-specific shifts
- [x] 4.4 Implement `orientation` mode with stage-proportional off-axis group-specific shifts
- [x] 4.5 Implement `shape` mode with non-monotone stage-specific group shifts
- [x] 4.6 Validate that `shape` mode requires at least three stages
- [x] 4.7 Record trajectory mode, effect size, and effect vectors or coefficients in truth metadata

## 5. Generator Entry Points

- [x] 5.1 Implement a pure transformation function that accepts an `InterSIMResult`
- [x] 5.2 Implement a convenience function that accepts InterSIM parameters plus generator parameters and calls `run_intersim()`
- [x] 5.3 Ensure InterSIM bridge errors propagate clearly from the convenience function
- [x] 5.4 Ensure returned omics matrices remain row-aligned to sample metadata after effect injection

## 6. Tests

- [x] 6.1 Add pure Python fixture builders for small InterSIM-like results
- [x] 6.2 Test clusters-as-stages mapping and preservation of original cluster labels
- [x] 6.3 Test deterministic within-stage group assignment
- [x] 6.4 Test validation errors for insufficient stage size and invalid shape mode
- [x] 6.5 Test explicit and proportion-based affected-feature selection
- [x] 6.6 Test `none` and zero-effect scenarios preserve omics values
- [x] 6.7 Test each non-null trajectory mode changes only affected features in the intended group/stage pattern
- [x] 6.8 Test convenience generation using mocks for `run_intersim()`

## 7. Documentation and Verification

- [x] 7.1 Document the clusters-as-stages assumption and output dataset contract
- [x] 7.2 Document supported trajectory modes and expected interpretation
- [x] 7.3 Run `uv run pytest tests/ -m "not slow" --tb=short`
- [x] 7.4 Run `uv run ruff check src/ tests/`
- [x] 7.5 Run `uv run mypy src/motco/`
