## 1. Optimize `_estimate_orientation`

- [ ] 1.1 In `src/motco/stats/trajectory.py`, replace the `C = X.T @ X; eigh(C)` block with `_, _, Vt = np.linalg.svd(X, full_matrices=False); orientation = Vt[0, :]`
- [ ] 1.2 Verify sign-flip logic is unchanged (same `c1 = float(orientation @ X[0, :])` check)

## 2. Optimize `_estimate_shape` GPA inner loop

- [ ] 2.1 Add a branch in the GPA loop: `if n_dimensions > n_levels:` (thin path) `else:` (original full path, unchanged)
- [ ] 2.2 Implement the thin path:
  - QR decompositions: `Q1, R1 = np.linalg.qr(Z1.T, mode='reduced')` and same for `Z2`
  - Small matrix: `K = R2 @ R1.T` (k×k)
  - SVD: `U_k, S_k, Vt_k = np.linalg.svd(K, full_matrices=True)`
  - `sig = 1.0` (det of p×p rank-k matrix is 0 when p > k)
  - `beta = float(S_k.sum())` (all k SVs positive)
  - `inner = R1.T @ (Vt_k.T @ U_k.T)` (k×k)
  - `temp2[i] = beta * (inner @ Q2.T)` (k×p)
- [ ] 2.3 Remove the `detH = np.linalg.det(H)` call from the thin path (it no longer exists)

## 3. Add numerical equivalence test

- [ ] 3.1 In `tests/test_trajectory.py`, add `test_gpa_thin_matches_full` parametrized over `(n_levels, n_dimensions)` pairs: `(4, 658)`, `(4, 10)`, `(4, 3)` (the p < k edge case)
- [ ] 3.2 For each pair: construct random (2*n_levels × n_dimensions) observed-vectors and a two-group contrast; call `_estimate_shape` with the current code (full path) to get reference, then call it again after the patch and assert `np.allclose(result, reference, atol=1e-10)`
- [ ] 3.3 Add `test_orientation_thin_matches_full` parametrized over the same (k, p) pairs: call `_estimate_orientation` on a random (k × p) matrix and verify output matches a reference computed via the `eigh` path

## 4. Verify and gate

- [ ] 4.1 `uv run ruff check src/ tests/` — no new errors
- [ ] 4.2 `uv run mypy src/motco/` — no new errors
- [ ] 4.3 `MOTCO_TEST_PERMS=99 uv run pytest tests/ -m "not slow" --tb=short` — all tests pass including the new equivalence tests
- [ ] 4.4 Quick smoke timing: run `estimate_difference` on a (120 × 658) concat-space matrix with 49 RRPP permutations and confirm it completes in < 60s (vs the previous ~628s)
