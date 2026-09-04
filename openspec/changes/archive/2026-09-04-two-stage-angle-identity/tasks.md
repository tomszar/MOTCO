## 1. Fast identity tests (tests/test_trajectory_orientation.py)

- [x] 1.1 Add a synthetic two-stage identity test: generate several pairs of two-stage trajectories (varied dimension, including a near-0° and a >90° pair), compute the pairwise angle via the existing `_angle_between` production-path helper, and assert `cos(angle)` equals the inner product of the unit transition vectors at `atol=1e-12`; verify with `uv run pytest tests/test_trajectory_orientation.py -v`.
- [x] 1.2 Add an example1 fixture identity test: load `evo_649_sm_example1.csv`, rebuild the design (PCA to 2 components, full model, sorted group/level order, as in the slow regression test) without RRPP, run `estimate_difference`, and assert each pairwise angle's cosine matches the direct transition-vector angle derived from the same fitted LS means at `atol=1e-12`; include a coarse sanity assertion that the t1/t3 and t2/t3 angles are ≈74.70/76.49; verify with `uv run pytest tests/test_trajectory_orientation.py -v`.
- [x] 1.3 In the same fixture test (or a sibling), assert the R artifact relation against `results_example1.csv`: each committed angle equals the direct-vector angle or its supplement (`min(|θ−exp|, |180−θ−exp|) < 1e-3`), with a comment naming the raw sign anchor (`evo_649_sm_suppmat.r:64`) as the cause; verify with `uv run pytest tests/test_trajectory_orientation.py -v`.

## 2. Tighten the slow regression test (tests/test_permutation.py)

- [x] 2.1 In `test_example1_expected_results_match`, compute the expected angle per pair from the fitted LS means' unit transition vectors and replace the `angle_ok = isclose(ang, exp) or isclose(ang, 180 − exp)` acceptance with a single comparison against that expectation (keep the CSV's angle for the θ-or-supplement relation check and the p-value comparisons unchanged); verify the diff no longer contains a `180.0 - exp_angle` acceptance path.
- [x] 2.2 Run the tightened slow test once to confirm it passes: `MOTCO_TEST_PERMS=99 uv run pytest tests/test_permutation.py::test_example1_expected_results_match -v` (low-perm smoke; angle expectations are permutation-free).

## 3. Gate

- [x] 3.1 Run the pre-commit gate and confirm all three pass: `uv run ruff check src/ tests/ && uv run mypy src/motco/ && MOTCO_TEST_PERMS=99 uv run pytest tests/ -m "not slow" --tb=short`.
