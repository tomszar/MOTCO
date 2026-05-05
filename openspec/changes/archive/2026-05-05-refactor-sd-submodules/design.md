## Context

`sd.py` (876 lines) bundles three logically independent concerns with a clean one-way dependency chain: design matrix construction ← trajectory estimation ← RRPP permutation. No circular dependencies exist, making the split straightforward. All import sites are internal to the repository.

Current import graph:
```
stats/__init__.py  →  sd.py  (all public symbols)
cli.py             →  sd.py  (RRPP, estimate_betas, estimate_difference)
test_*.py          →  sd.py  (various symbols)
```

Target import graph:
```
stats/__init__.py  →  design.py, trajectory.py, permutation.py
cli.py             →  motco.stats  (flat namespace, no direct submodule imports)
test_design.py     →  design.py
test_trajectory.py →  trajectory.py
test_permutation.py→  permutation.py
```

## Goals / Non-Goals

**Goals:**
- Split `sd.py` into `design.py`, `trajectory.py`, `permutation.py` with no underscore prefix (public submodules)
- Mirror the module split in the test suite
- Preserve the flat `motco.stats.*` public API exactly
- Delete `sd.py` with no shim or compatibility layer

**Non-Goals:**
- Changing any function signatures, behavior, or algorithms
- Splitting `test_validation.py` pls/snf tests (tracked in issue #15)
- Adding new functionality

## Decisions

### Decision: Public submodule names (no leading underscore)

`design.py`, `trajectory.py`, `permutation.py` — not `_design.py` etc.

**Rationale**: Leading underscores signal "do not import directly." Since the goal is for submodules to be the real API (not a hidden implementation detail behind a facade), public names are appropriate. Users who want finer-grained imports can use `from motco.stats.trajectory import estimate_difference` directly.

**Alternative considered**: `_design.py` / `_trajectory.py` / `_permutation.py` with `sd.py` as a permanent re-export facade. Rejected because it keeps `sd.py` alive as dead weight and discourages direct submodule imports.

### Decision: Delete `sd.py` immediately, no transition shim

All affected import sites are internal. The "breaking" change only affects `motco.stats.sd.*` direct imports, all of which live in `cli.py` and test files within this repo. These are updated in the same change.

**Alternative considered**: Keep `sd.py` as a thin re-export shim with a deprecation warning. Rejected because there are no external callers — the shim would exist only to be deleted later, adding pointless complexity.

### Decision: cli.py imports from `motco.stats` flat namespace

Rather than updating `cli.py` to import from the new submodules directly, import from `motco.stats` (the existing flat namespace). This keeps `cli.py` decoupled from internal module structure.

### Decision: Symbol allocation across submodules

| Module | Symbols |
|---|---|
| `design.py` | `center_matrix`, `get_model_matrix`, `build_ls_means` |
| `trajectory.py` | `pair_difference`, `estimate_difference`, `estimate_betas`, `get_observed_vectors`, `_estimate_size`, `_estimate_orientation`, `_estimate_shape`, `_OPA` |
| `permutation.py` | `_RRPPWorker`, `RRPP` |

`_estimate_size`, `_estimate_orientation`, `_estimate_shape`, `_OPA` stay private (single underscore) within `trajectory.py` — they are implementation helpers not intended for direct use.

### Decision: Test file mapping

| New test file | Sources | Coverage |
|---|---|---|
| `test_design.py` | `test_sd_smoke.py` (2 tests) + `test_validation.py` (5 tests) | `design.py` |
| `test_trajectory.py` | `test_sd_smoke.py` (1 test) + `test_validation.py` (5 tests) | `trajectory.py` |
| `test_permutation.py` | `test_sd_expected_example1.py` + `test_sd_expected_example2.py` + `test_validation.py` (2 tests) | `permutation.py` |

`test_sd_smoke.py`, `test_sd_expected_example1.py`, `test_sd_expected_example2.py` are deleted. `test_validation.py` retains only pls/snf tests.

## Risks / Trade-offs

- **Breaking `motco.stats.sd` imports** → Mitigation: all sites are internal and updated in the same PR. No external callers identified.
- **Test coverage during migration** → Mitigation: new test files are created before old ones are deleted; CI runs on the complete set.
- **Merge conflicts if `sd.py` is edited concurrently** → Low risk; no active feature branches touching `sd.py` at time of writing.

## Migration Plan

1. Create `design.py`, `trajectory.py`, `permutation.py` from `sd.py` content
2. Update `stats/__init__.py` imports
3. Create `test_design.py`, `test_trajectory.py`, `test_permutation.py`
4. Update `cli.py` and `test_cli.py` imports
5. Remove sd-related tests from `test_validation.py`
6. Delete `sd.py`, `test_sd_smoke.py`, `test_sd_expected_example1.py`, `test_sd_expected_example2.py`
7. Run full test suite to confirm no regressions

Rollback: revert the PR — no data migrations or external state involved.
