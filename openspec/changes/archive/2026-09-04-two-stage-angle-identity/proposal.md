## Why

At exactly two stages, a trajectory's leading principal axis is mathematically identical to its transition vector, so the pairwise `angle` from `estimate_difference` must equal the angle between the groups' unit transition vectors — an identity, not an approximation. This identity is currently enforced only at the estimator level (single-trajectory orientation vector); nothing tests it at the pairwise-angle level, and the example1 regression test accepts `angle ≈ exp OR 180 − exp`, an escape hatch that dates from before the net-displacement sign convention was settled and that would silently pass a sign-flip regression today.

## What Changes

- Add a spec requirement to `trajectory-orientation-invariance`: at exactly two stages, the pairwise `angle` equals the direct transition-vector angle as an identity (machine precision on the cosine).
- Add fast, permutation-free tests in `tests/test_trajectory_orientation.py`:
  - synthetic two-stage configurations: `estimate_difference` angle vs. `arccos` of the normalized transition-vector dot product, asserted as an identity;
  - the `evo_649_sm_example1.csv` fixture: expected angles derived from the data's own transition vectors (74.70 / 76.49 for t1/t3 and t2/t3), with R's committed 105.30 / 103.51 checked as the documented raw sign-anchor artifact (each committed angle equals θ or 180 − θ).
- Tighten the slow example1 regression test in `tests/test_permutation.py` to drop the `angle ≈ exp OR 180 − exp` acceptance in favor of the direct-vector-derived expectation.
- No production code changes: `_estimate_orientation` and `estimate_difference` keep the single PC1 code path (matching the reference supplement, which also never branches on stage count).

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `trajectory-orientation-invariance`: add a requirement that the two-stage pairwise `angle` equals the direct transition-vector angle (identity contract at the `estimate_difference` level, including reproduction on the committed example1 fixture and documentation of the R sign-anchor artifact). The existing estimator-level two-stage scenario is unchanged.

## Impact

- `openspec/specs/trajectory-orientation-invariance/spec.md` — one new requirement (via delta spec).
- `tests/test_trajectory_orientation.py` — new fast identity tests.
- `tests/test_permutation.py` — `test_example1_expected_results_match` loses its supplementary-angle escape hatch (slow suite only; expected values unchanged in substance, just pinned to MOTCO's convention).
- No changes to `src/motco/` — test-only change; RRPP null distributions and all production outputs are untouched.
