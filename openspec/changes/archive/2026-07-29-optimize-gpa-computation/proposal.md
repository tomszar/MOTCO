## Why

`_estimate_shape` in `stats/trajectory.py` runs an iterative GPA (Generalized Procrustes Analysis) that, in each iteration, computes the SVD of the (p × p) cross-covariance matrix `H = Z2.T @ Z1` (where p = n_dimensions, the latent space size). For concat integration, p = 658 and every GPA iteration costs ~57ms just for the SVD — totalling ~628s for a single 49-permutation RRPP call.

The same structural problem appears in `_estimate_orientation`, which forms the (p × p) sample covariance matrix `X.T @ X` and calls `eigh`.

Both functions are O(p³) per call. Since Z1, Z2 are (k × p) matrices with k = n_levels (always small, e.g. 4), the rank of H is at most k, and the full p × p computation is pure waste: the entire rotation only depends on the k-dimensional row space. A thin-QR-based path reduces both to O(p k²), delivering a measured ~632× speedup.

**Root cause history**: this bottleneck has been present since the first commit. A `5d18f42` vectorisation pass added a batched einsum that *looked* like an optimisation but produced an H of shape (G, p, p) — the SVD cost was unchanged. The current code (after the `de031fa` refactor) reverts to an explicit per-group loop with the correct pPsup Procrustes convention, still O(p³).

**Scope note**: the bottleneck is only catastrophic for `concat` integration (p = 658). Production methods (PLS: p ≈ 2–5; SNF: p = 10) are already fast. However, concat is used as the reference baseline and in smoke tests, so fixing it makes the full suite tractable.

## What Changes

- **`_estimate_shape`**: replace `svd(H p×p)` + `det(H)` with a thin-QR path that forms a (k × k) matrix and runs `svd` on that.
- **`_estimate_orientation`**: replace `eigh(X.T @ X)` on the (p × p) covariance matrix with `svd(X)` on the (k × p) data matrix directly.
- **New regression test**: a parametrised test that runs both the old (full) and new (thin) code paths on the same input and asserts that their outputs match to numerical precision, for both small (p < k) and large (p >> k) cases.

## Non-Goals

- No change to the mathematical semantics of the Procrustes alignment.
- No change to public API (`estimate_difference`, `_estimate_shape`, `_estimate_orientation` signatures stay the same).
- `_OPA` is dead code (not called by any production path); it will not be touched.

## Capabilities Modified

- `stats/trajectory.py` internal helpers — performance-only change; all outputs remain numerically equivalent.
