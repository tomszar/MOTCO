## Why

MOTCO production `shape` should mean strict geometric-morphometric shape: trajectory configuration remaining after translation, rigid rotation, and uniform scale are removed. Recent inverse-PLS diagnostics showed a nonzero `shape` response for a known rigid latent rotation, so the current estimator may not satisfy the intended scientific contract.

## What Changes

- Audit the current `_estimate_shape` GPA/Procrustes implementation using minimal deterministic trajectory configurations with known translation, scale, rotation, reflection, and genuine bend relationships.
- Compare the current leave-one-out GPA behavior against direct pairwise Procrustes expectations and the legacy R reference procedure.
- Define and enforce production invariants for `shape`: translation, uniform scaling, and rigid rotation MUST yield zero shape distance within numerical tolerance.
- Preserve `delta` as magnitude and `angle` as orientation; ensure `shape` no longer absorbs pure magnitude or orientation differences when strict morphometric shape is unchanged.
- Document the reflection policy explicitly: whether mirror images are removed by the alignment or treated as distinct configurations.
- If the current estimator violates the contract, update implementation and tests; if legacy behavior differs from strict morphometric shape, document the legacy divergence in the audit findings.

## Capabilities

### New Capabilities

- `trajectory-shape-invariance`: Defines the strict geometric-morphometric invariance contract for MOTCO trajectory `shape` and the diagnostic outputs needed to verify it.

### Modified Capabilities

- None.

## Impact

- Affected code: `src/motco/stats/trajectory.py`, especially `_estimate_shape` and supporting Procrustes helpers.
- Affected tests: new deterministic invariance/regression tests in `tests/test_trajectory.py` or a focused shape test module.
- Affected documentation: trajectory-analysis docs and roadmap language describing `shape` semantics and any legacy estimator divergence.
- No new runtime dependencies are expected; SciPy may be used in tests as an installed reference implementation if useful.
