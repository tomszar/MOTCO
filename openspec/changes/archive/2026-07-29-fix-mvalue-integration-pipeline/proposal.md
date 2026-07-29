## Why

The rung ladder investigation (see `openspec/changes/rung-ladder-conclusion.md`) established that leaving methylation in beta (B-value) space at the integration step manufactures magnitude→orientation cross-talk because the sigmoid nonlinearity breaks direction-preservation: `sigmoid(c·a) ≠ c·sigmoid(a)`. Every integration method (concat, SNF, PLS) inherits this artifact regardless of its own linear-algebraic cleanliness. The fix — convert methylation back to M-value (logit) space before integration — resolves the root cause, not a symptom.

## What Changes

- **Add `logit(x, clip)` to `generator.py`** as the canonical companion to `rev_logit`; this is where the inverse-logit transform already lives and the natural home for its inverse.
- **Apply `logit` to the methylation layer in all three integration helpers** (`_concat_integration`, `_snf_integration`, `_pls_integration` in `evaluation.py`) before any standardisation or model fitting.
- **Remove the local `beta_to_mvalue` definition from `methylation_recovery.py`** and replace it with an import of the newly canonical `logit` from `generator.py`, keeping the public alias.
- **Add `examples/trajectory_power_study/study.json`** — the paper-grade PLS study config referenced in the README but currently missing.

No breaking API changes; `SemiSyntheticTrajectoryDataset.methylation` continues to store B values (the conversion is applied at the point of consumption).

## Capabilities

### New Capabilities

- `mvalue-integration`: M-value conversion applied to methylation at the integration step — the contract that integration always operates in the generator's native M-value space.

### Modified Capabilities

- `simulation-evaluation-harness`: Integration functions now apply a clipped logit to methylation before standardisation/fitting. The integration result is functionally different (M-value input, not B-value) for all methods.

## Impact

- **`src/motco/simulations/generator.py`** — add `logit` function.
- **`src/motco/simulations/evaluation.py`** — three `_*_integration` helpers updated.
- **`src/motco/simulations/methylation_recovery.py`** — `beta_to_mvalue` becomes an imported alias.
- **`examples/trajectory_power_study/study.json`** — new file.
- Tests that snapshot integration outputs will need their reference values regenerated (the methylation input changes); no test logic changes.