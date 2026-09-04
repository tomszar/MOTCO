## Context

See proposal.md — Why. Relevant current state:

- `_estimate_orientation` (`src/motco/stats/trajectory.py:535`) computes PC1 of the centered stage configuration via SVD and signs it by net displacement (`PC1 · (stage_last − stage_first)`). At two stages the centered configuration is `±(v/2)` for `v = stage₂ − stage₁`, rank one, so PC1 is exactly `v/‖v‖`.
- The estimator-level two-stage scenario ("orientation is the unit transition vector") is already spec'd and tested (`tests/test_trajectory_orientation.py:142`). Nothing tests the pairwise-angle level (`estimate_difference`'s dot → clip → arccos path).
- `test_example1_expected_results_match` (`tests/test_permutation.py:100`, `@pytest.mark.slow`) accepts `angle ≈ exp OR 180 − exp` against `results_example1.csv`. R's committed 105.30/103.51 (t1/t3, t2/t3) are supplements of the direct-vector angles 74.70/76.49 because R's sign anchor (`evo_649_sm_suppmat.r:64`) depends on position relative to the PCA origin; t1/t2 (1.79°) matches directly. The deviation is documented in the `_estimate_orientation` docstring.
- The `_angle_between` helper in `tests/test_trajectory_orientation.py` already routes synthetic configurations through the production `estimate_difference` path with identity model/LS matrices.

## Goals / Non-Goals

**Goals:**
- Enforce the two-stage angle identity at the `estimate_difference` level, on synthetic data and on the committed example1 fixture, permutation-free and in the fast suite.
- Pin the example1 expected angles to the progression convention everywhere, removing the supplementary-angle acceptance from the slow regression test.

**Non-Goals:**
- No changes to `src/motco/` — the single PC1 code path stays for all stage counts (the R reference likewise never branches on stage count).
- No changes to `results_example1.csv` or any committed fixture data.
- No changes to the example2 (5-level) regression test: all sign anchors agree on that geometry, so its comparisons are not affected by the artifact this change pins down.
- No treatment of the zero-displacement degeneracy beyond its existing documentation.

## Decisions

1. **Assert on the cosine, not the angle, for the identity tests.** `arccos` has derivative `−1/√(1−x²)`: well-conditioned at generic angles but degrading to `√ε` sensitivity near 0° and 180°. Comparing `cos(reported angle)` against the unit-transition-vector inner product at `atol=1e-12` states the identity uniformly across the angle range. Alternative — degree-space comparison at 1e-8 — rejected as angle-dependent and looser than the identity warrants.

2. **Derive example1 expected angles from the data, not from hard-coded constants.** The fast fixture test fits the same LS means as the pipeline, computes unit transition vectors per group, and asserts `estimate_difference` matches `arccos` of their inner products. The values 74.70/76.49 appear only in a sanity assertion (coarse tolerance) documenting the magnitude, so the test is self-validating rather than dependent on retyped constants. The relation to R's CSV is asserted separately: each committed angle equals θ or 180 − θ (min-of-both comparison at the CSV's precision), documenting the artifact without endorsing it.

3. **Tighten the slow test by replacing its expectation, not its structure.** In `test_example1_expected_results_match`, replace `angle_ok = isclose(ang, exp) or isclose(ang, 180 − exp)` with a comparison against the direct-vector-derived expected angle (computed from the same fitted LS means in the test). p-value comparisons are untouched — the RRPP angle statistic is two-sided-symmetric under the permutation scheme only through its observed value, and the committed p-values (0.001) already match under either convention.

4. **New tests live in `tests/test_trajectory_orientation.py`.** That module owns the orientation invariance contract and already has the `_angle_between` production-path helper; the fixture test adds a small loader for `evo_649_sm_example1.csv` mirroring the slow test's design construction (PCA to 2 components, full model, sorted level order) but without RRPP.

## Risks / Trade-offs

- [The fast fixture test duplicates the slow test's design-construction boilerplate] → Keep it minimal (a local helper); sharing via conftest is possible later if a third consumer appears.
- [1e-12 cosine tolerance could flake if BLAS/SVD implementations differ across platforms] → Both compared quantities pass through the same LS-mean fit; the only divergent operations are one SVD of a 2×k rank-1 matrix vs. one normalization — both accurate to a few ulp. CI runs the fast suite on the pinned lockfile; loosen to 1e-10 only if evidence appears.
- [Dropping the supplementary acceptance makes the slow test stricter than R's committed CSV] → Intentional: the CSV's angle column is retained for the θ-vs-180−θ relation and p-values, while the angle expectation itself comes from the data under MOTCO's documented convention.
