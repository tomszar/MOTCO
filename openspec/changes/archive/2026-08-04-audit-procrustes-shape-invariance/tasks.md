## 1. Diagnostic Fixtures

- [x] 1.1 Add deterministic non-collinear 2D trajectory fixtures with at least three stages and known translation, uniform-scale, proper-rotation, reflection, and bend transforms.
- [x] 1.2 Add helper assertions for zero-within-tolerance shape distance and positive residual shape distance.
- [x] 1.3 Add diagnostic tests that record current `_estimate_shape` behavior for each fixture before changing estimator semantics.

## 2. Reference Comparisons

- [x] 2.1 Implement or test a direct pairwise Procrustes reference calculation using centering, centroid-size scaling, and proper orthogonal rotation.
- [x] 2.2 Compare current leave-one-out GPA behavior against the direct pairwise reference for every deterministic fixture.
- [x] 2.3 Compare against the legacy R `pgpa`/`pPsup` procedure when the R environment is available, without making R the production oracle.

## 3. Shape Estimator Contract

- [x] 3.1 Update the production shape estimator so translated trajectories have zero `shape` distance within tolerance.
- [x] 3.2 Update the production shape estimator so uniformly scaled trajectories have zero `shape` distance within tolerance.
- [x] 3.3 Update the production shape estimator so proper rigid rotations have zero `shape` distance within tolerance.
- [x] 3.4 Preserve positive `shape` distance for genuine residual configuration changes such as an interior-stage bend.
- [x] 3.5 Enforce and document the default reflection policy with a deterministic reflected-trajectory test.

## 4. Integration and Regression Coverage

- [x] 4.1 Add `estimate_difference` regression tests showing pure magnitude contrasts may affect `delta` but not `shape`.
- [x] 4.2 Add `estimate_difference` regression tests showing pure orientation contrasts may affect `angle` but not `shape`.
- [x] 4.3 Add permutation/RRPP coverage or focused regression checks for corrected shape availability and pairwise matrix output.
- [x] 4.4 Update or replace tests that only assert parity with the legacy GPA implementation where that parity conflicts with strict shape semantics.

## 5. Documentation and Findings

- [x] 5.1 Update trajectory-analysis documentation to define `shape` as strict morphometric residual configuration after translation, proper rotation, and uniform scale are removed.
- [x] 5.2 Add a compact audit findings note summarizing current legacy behavior, corrected behavior, reflection policy, and before/after deterministic cases.
- [x] 5.3 Update the roadmap to mark shape invariance as resolved or to record any remaining blocker.

## 6. Verification

- [x] 6.1 Run focused trajectory and permutation tests covering the shape estimator.
- [x] 6.2 Run `uv run pytest tests/ -m "not slow" --tb=short` with the project runtime environment.
- [x] 6.3 Run `uv run ruff check src/ tests/` and `uv run mypy src/motco/`.
- [x] 6.4 Validate the OpenSpec change with `openspec validate audit-procrustes-shape-invariance --strict`.
