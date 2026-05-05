## Why

`src/motco/stats/sd.py` has grown to 876 lines bundling three unrelated concerns — design matrix construction, trajectory estimation, and RRPP permutation — making it hard to navigate, test, and extend. Splitting it into focused submodules reduces cognitive load and enables targeted unit testing of each layer independently.

## What Changes

- **New**: `src/motco/stats/design.py` — design matrix and LS-means utilities (`center_matrix`, `get_model_matrix`, `build_ls_means`)
- **New**: `src/motco/stats/trajectory.py` — trajectory estimation and geometric metrics (`pair_difference`, `estimate_difference`, `estimate_betas`, `get_observed_vectors`, and private helpers)
- **New**: `src/motco/stats/permutation.py` — RRPP permutation infrastructure (`_RRPPWorker`, `RRPP`)
- **Modified**: `src/motco/stats/__init__.py` — update imports to draw from the three new submodules instead of `sd.py`
- **Removed**: `src/motco/stats/sd.py` — deleted after all imports are migrated (**BREAKING** for direct `motco.stats.sd.*` imports)
- **New**: `tests/test_design.py`, `tests/test_trajectory.py`, `tests/test_permutation.py` — test files mirroring the new module structure
- **Removed**: `tests/test_sd_smoke.py`, `tests/test_sd_expected_example1.py`, `tests/test_sd_expected_example2.py` — replaced by the new test files
- **Modified**: `tests/test_validation.py` — sd-related validation tests moved to the new test files; only pls/snf tests remain
- **Modified**: `src/motco/cli.py`, `tests/test_cli.py` — import paths updated

## Capabilities

### New Capabilities

None — this is a pure internal restructuring. No new behavior is introduced.

### Modified Capabilities

None — existing specs (`api-reference`, `rrpp-progress`) remain accurate. No requirement-level behavior changes.

## Impact

- **Breaking**: Direct imports from `motco.stats.sd` (e.g., `from motco.stats.sd import RRPP`) will break. The public `motco.stats` namespace is unaffected.
- **Internal only**: All affected import sites (`cli.py`, five test files) are within this repository.
- **No API change**: The flat `motco.stats.*` namespace exposed via `stats/__init__.py` is unchanged.
