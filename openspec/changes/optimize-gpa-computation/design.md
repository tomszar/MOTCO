# Design: Thin-QR Procrustes Optimization

## Mathematical Derivation

### Why `sig = +1` always when `p > k`

The reflection-correction factor `sig` in the current code is:

```python
detH = float(np.linalg.det(H))   # H is p×p
sig = -1.0 if detH < 0.0 else 1.0
```

When `p > k`, `rank(H) = rank(Z2.T @ Z1) ≤ k < p`, so `det(H) = 0` and `sig = +1` always. This is correct: in the high-dimensional case there is no unique notion of "reflection" in the (p − k)-dimensional null space, and the code already defaults to no reflection. The thin path must replicate this — **do not substitute `det(K_small)` for `det(H)`** (that was the error in the earlier failed attempt, which gave `sig = −1` and wrong beta).

---

### `_estimate_orientation` — replace `eigh(p×p)` with `svd(k×p)`

**Current**: form `C = X.T @ X / (n−1)` (p×p), call `eigh(C)`, take last eigenvector.

**Optimized**: the leading eigenvector of `X.T @ X` is the right singular vector of `X` for the largest singular value.

```
X (k×p)  →  svd(X, full_matrices=False)  →  U (k×k), S (k,), Vt (k×p)
orientation = Vt[0, :]   # largest singular value is index 0 (descending)
```

Complexity: O(k² p) vs O(p³). No change to semantics or sign convention.

---

### `_estimate_shape` — thin-QR path for `pPsup`

**Setup**: `Z1`, `Z2` are (k×p) centered-and-scaled shapes, k = n_levels (e.g. 4), p = n_dimensions (e.g. 658).

**Step 1 — Economy QR of the transposes**:

```
Z1.T = Q1 R1   (economy QR: Q1 is p×k with orthonormal columns, R1 is k×k)
Z2.T = Q2 R2   (economy QR: Q2 is p×k with orthonormal columns, R2 is k×k)
```

`np.linalg.qr(Z1.T, mode='reduced')` returns `(p×k, k×k)` when `p > k`.

**Step 2 — Factor H through the small matrix**:

```
H = Z2.T @ Z1
  = (Q2 R2) @ (R1.T Q1.T)   [since Z1 = R1.T Q1.T]
  = Q2 K Q1.T                where K = R2 @ R1.T  (k×k)
```

**Step 3 — SVD of the small matrix**:

```
K = U_k S_k Vt_k   (all k×k)

→ thin SVD of H:
    U_H  = Q2 U_k     (p×k, orthonormal columns)
    S_H  = S_k         (k,)
    V_H  = Q1 Vt_k.T  (p×k, orthonormal columns)
```

**Step 4 — Rotation and aligned shape** (with `sig = +1`):

```
Γ = V_H U_H.T = Q1 Vt_k.T U_k.T Q2.T

Z1 @ Γ = (R1.T Q1.T) @ Q1 Vt_k.T U_k.T Q2.T
        = R1.T @ Vt_k.T @ U_k.T @ Q2.T
```

The last line uses `Q1.T @ Q1 = I_k` (economy QR, not `I_p`).

**Step 5 — beta** (with `sig = +1`):

```
beta = sum(S_k)   # all k singular values positive
```

**Implementation**:

```python
Q1, R1 = np.linalg.qr(Z1.T, mode='reduced')          # (p×k), (k×k)
Q2, R2 = np.linalg.qr(Z2.T, mode='reduced')          # (p×k), (k×k)
K = R2 @ R1.T                                          # (k×k)
U_k, S_k, Vt_k = np.linalg.svd(K, full_matrices=True) # all k×k
beta = float(S_k.sum())
inner = R1.T @ (Vt_k.T @ U_k.T)                       # (k×k)
temp2[i] = beta * (inner @ Q2.T)                       # (k×p)
```

**Complexity comparison**:

| Step                  | Old (full)         | New (thin)        | Ratio (p=658, k=4) |
|-----------------------|--------------------|-------------------|--------------------|
| Form H / QR           | O(k p²)            | O(p k²)           | p/k = 164×         |
| SVD                   | O(p³)              | O(k³)             | (p/k)³ = 4.4M×     |
| det(H)                | O(p³)              | skipped (sig=+1)  | —                  |
| Rotate shape          | O(k p²) (k p matmul) | O(k² p)         | p/k = 164×         |

Dominant cost shifts from `svd(658×658)` (~57ms) to `qr(658×4)` (~0.09ms per pair) — **~632× speedup per GPA iteration** as benchmarked.

---

### Edge case: `p ≤ k`

When `n_dimensions ≤ n_levels` (unusual but possible in tests with synthetic low-dimensional data):
- Economy QR of `Z1.T` (p×k) with `p ≤ k` gives `Q1` (p×p), `R1` (p×k) — `K = R2 @ R1.T` becomes (p×k)(k×p) = (p×p), no savings.
- More importantly, `det(H)` can be non-zero and meaningful, so `sig = +1` is wrong.
- **Guard**: `if n_dimensions > n_levels: # thin path else: # original path`.

---

## Test Plan

### Numerical equivalence test (`tests/test_trajectory.py`)

New test `test_gpa_thin_matches_full`:

- Construct two random (k×p) centered-scaled matrices for several (k, p) pairs:
  - `(4, 658)` — production case, p >> k
  - `(4, 10)` — intermediate
  - `(4, 3)` — p < k (original path should be used; thin path falls back)
- Run both the full-SVD path (extracted as a helper or tested via `_estimate_shape` with a mocked n_dimensions) and the optimized path.
- Assert `np.allclose(result_full, result_thin, atol=1e-10)`.

The cleanest approach: extract the current inner loop body as a `_ppsup_full(Z1, Z2)` helper (for testing only), then test both against `_estimate_shape` output. Alternatively, pass the two shapes through `estimate_difference` with a 2-group, k-level dataset at both small and large p, and compare the shape distance matrix.

**Recommended**: parametrize `test_gpa_thin_matches_full` with `(n_levels, n_dimensions)` pairs, construct a `(2*n_levels, n_dimensions)` observed-vectors array and a two-group contrast, and call `_estimate_shape` before and after the patch. Since the function is deterministic given the same input, the test proves equivalence.
