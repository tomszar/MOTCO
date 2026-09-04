## 1. Shared orientation functional

- [x] 1.1 Expose the orientation estimator from `stats/trajectory.py` for in-package reuse (public wrapper or documented private import per design D1) and verify a new unit test shows `attribution`'s functional and `_estimate_orientation` return identical vectors on the same configuration
- [x] 1.2 Add the degeneracy inputs: compute per-configuration relative eigengap via `configuration_spectrum` for observed and reconstructed group means, and verify a unit test pins the eigengap of a straight (≈1.0), bent, and near-isotropic configuration

## 2. Principal-orientation component in attribution

- [x] 2.1 Add `PrincipalOrientationAttribution` dataclass and the `principal_orientation` result field (observed / PLS-captured / residual contrasts, per-group signed axes, per-configuration eigengaps, degeneracy flag, availability flags), constructed after the existing transition build; verify `MOTCO_TEST_PERMS=99 uv run pytest tests/test_attribution.py -v` passes with the new field populated
- [x] 2.2 Wire `eigengap_threshold` (default 0.05) and the reserved `principal` identifier into `AttributionConfig`, including collision validation against stage-derived transition ids; verify tests cover the default, an override, and the collision error
- [x] 2.3 Handle degeneracies: eigengap below threshold sets the flag (contrast still returned); net displacement below `zero_tolerance` marks that group's principal orientation unavailable; verify unit tests for both paths, including a closed trajectory
- [x] 2.4 Route the principal contrast through the original-units conversion path and verify a test shows original-unit effects agree with the standardized ones under the supplied scales

## 3. Equivalence and consistency contracts

- [x] 3.1 Add the k = 2 equivalence test: principal-orientation contrasts (all three components) equal the single-transition contrasts exactly; verify it passes without tolerance relaxation beyond machine precision
- [x] 3.2 Add a k ≥ 3 consistency test on a bent trajectory: the principal contrast differs from every per-transition contrast while matching `_estimate_orientation` per group; verify with a fixed-seed fixture

## 4. Bootstrap stability

- [x] 4.1 Extend `_bootstrap` to compute the per-replicate principal contrast with per-replicate net-displacement sign anchoring, feeding sign/rank/top-k stability summaries keyed by the reserved identifier; verify the reproducibility test (same seed ⇒ identical summaries) covers the principal rows
- [x] 4.2 Cover replicate-level degeneracy: replicates with near-zero net displacement count as unavailable, mirroring zero-transition replicates; verify with a constructed near-closed fixture

## 5. Output surfaces and downstream consumers

- [x] 5.1 Add the principal-orientation frame to `attribution_frames` and `write_attribution_outputs` (flag and eigengaps as columns) without changing existing frame schemas; verify schema tests for old and new frames pass
- [x] 5.2 Update the interpretation metadata to state which quantity each decomposition explains (principal = tested `angle` estimand; transitions = per-step description, coincident only at k = 2); verify a test asserts the statement fields
- [x] 5.3 Audit `simulations/attribution_diagnostics.py` and any other `OrientationAttributionResult` consumers for additive compatibility (design risk 5); verify `MOTCO_TEST_PERMS=99 uv run pytest tests/ -m "not slow" --tb=short` passes

## 6. Documentation alignment

- [x] 6.1 Update the `stats/attribution.py` module docstring and `analyze_orientation_attribution` docstring for the two decompositions and the degeneracy qualifier; verify docstrings render (mkdocs/pydoc) without errors
- [x] 6.2 Correct `docs/roadmap.md:133` from planned to delivered wording for signed principal orientations; verify by reading the diff
- [x] 6.3 At sync/archive time, edit the `trajectory-orientation-invariance` main-spec Purpose line to the principal-axis-divergence phrasing (design D7) and verify `openspec validate --strict` passes

## 7. Gate

- [x] 7.1 Run the pre-commit gate — `uv run ruff check src/ tests/ && uv run mypy src/motco/ && MOTCO_TEST_PERMS=99 uv run pytest tests/ -m "not slow" --tb=short` — and verify all three pass
