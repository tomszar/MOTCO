## Why

`derive_replicate_seed` (`src/motco/simulations/grid.py:280-288`) produces
unsigned 32-bit integers in `[0, 2³² − 1]`, but the InterSIM R bridge
hands those seeds to R's `set.seed()`, which only accepts signed 32-bit
values. Roughly half of all derived seeds have the high bit set and
silently become `NA` in R via `as.integer()`, aborting `set.seed()`
with "supplied seed is not a valid integer". In the trajectory power
study smoke run this killed **64 of 144 replicates (44 %)** — the same
loss rate will occur in any paper-grade run unless we fix the
derivation. The docstring already claims a "32-bit seed", so the
implementation is also at odds with its declared contract.

## What Changes

- Constrain `derive_replicate_seed` to return values in
  `[0, 2³¹ − 1]` (mask the parsed digest with `& 0x7FFFFFFF`). Update
  the docstring to "31-bit unsigned" and explain the R compatibility
  reason.
- Add a defensive bounds check in
  `intersim._build_rscript_command` (or equivalent boundary) that
  raises `InterSIMError` when `params.seed` falls outside R's
  signed-32-bit range. Converts a silent R-side `NA` coercion into a
  clear Python error for any future caller that bypasses the
  derivation helper.
- Bump `parameter_signature` with a `seed_derivation_version` field
  so shards completed under the buggy derivation are automatically
  invalidated on resume. Without this, the existing
  `(cell_id, replicate_index)` resume guard would silently skip
  broken replicates after the fix lands.

No public CLI or config changes. No breaking changes for users
starting fresh studies; in-flight studies on stale shards must be
re-run (the new signature triggers re-execution automatically).

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `simulation-grid-orchestration`: `derive_replicate_seed` must
  return R-compatible signed-32-bit seeds, and `parameter_signature`
  must include a derivation-version tag.
- `intersim-simulation-bridge`: the bridge must reject seeds outside
  R's signed-32-bit range with a clear Python error instead of
  letting R coerce them to `NA`.

## Impact

- **Source.**
  - `src/motco/simulations/grid.py` — `derive_replicate_seed`,
    `parameter_signature`.
  - `src/motco/simulations/intersim.py` — boundary check in
    `_build_rscript_command`.
- **Tests.**
  - `tests/simulations/test_grid.py` (or wherever derivation is
    tested) — masked-range and determinism assertions; signature-
    version regression.
  - `tests/simulations/test_intersim.py` — out-of-range seed raises
    `InterSIMError`.
- **Operational.** Any shard JSONL files produced before this change
  are invalidated by the signature bump and will be re-executed on
  resume. The trajectory-power-study smoke (`/tmp/motco-smoke`) and
  any in-flight cluster runs must be re-launched; the README's
  resume instructions still apply.
- **Out of scope.** The independent `magnitude`-mode design issue
  (per-stage shifts use a fixed direction vector and therefore also
  rotate trajectory orientation, inflating the `angle` statistic
  under `magnitude` perturbations) — file as a follow-on proposal
  against `semisynthetic-trajectory-generator`.
