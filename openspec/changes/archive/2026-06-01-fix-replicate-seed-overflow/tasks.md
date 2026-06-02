## 1. Source changes

- [x] 1.1 In `src/motco/simulations/grid.py`, mask the result of `derive_replicate_seed` with `& 0x7FFFFFFF` and update the docstring to "31-bit unsigned" with a one-line note on R compatibility.
- [x] 1.2 In `src/motco/simulations/grid.py`, add `"seed_derivation_version": 2` to the payload built inside `parameter_signature`.
- [x] 1.3 In `src/motco/simulations/intersim.py`, add a bounds check at the top of `_build_rscript_command` that raises `InterSIMError` when `params.seed` is outside `[-2**31, 2**31 - 1]`. Include the offending value and accepted range in the message.

## 2. Tests

- [x] 2.1 In `tests/simulations/test_grid.py` (or the file that currently covers `derive_replicate_seed`), add a test that brute-forces several `(cell_id, replicate_index)` payloads known to land in the high half of the unsigned 32-bit range and asserts the returned seed sits in `[0, 2**31 - 1]`.
- [x] 2.2 Add a determinism regression: for a fixed payload that produced `2_797_983_684` pre-fix, assert the post-fix value equals `2_797_983_684 & 0x7FFFFFFF == 650_500_036`.
- [x] 2.3 Add a signature regression: assert two cells whose only difference is the (mocked) seed-derivation version produce different `parameter_signature` outputs, and that signatures are stable across runs at a fixed version.
- [x] 2.4 In `tests/simulations/test_intersim.py`, add tests that `_build_rscript_command` raises `InterSIMError` for seeds `2**31` and `-2**31 - 1`, and succeeds for `2**31 - 1`, `0`, and `-2**31`.

## 3. Validation gate

- [x] 3.1 Run `uv run ruff check src/ tests/`.
- [x] 3.2 Run `uv run mypy src/motco/`.
- [x] 3.3 Run `MOTCO_TEST_PERMS=99 uv run pytest tests/ -m "not slow" --tb=short`.

## 4. Smoke verification (optional, only if R + InterSIM available)

- [x] 4.1 Delete `/tmp/motco-smoke` and re-run the local smoke loop from the trajectory power study README; confirm the failure rate drops to ~0 (no `InterSIMRuntimeError: ... supplied seed is not a valid integer`). _Verified: 0 seed-overflow errors (was 64/144); 32 remaining failures are the unrelated `shape` mode + 2-cluster smoke config issue, out of scope here._
- [x] 4.2 Inspect `merged.jsonl` for a sample of `intersim_seed` values; confirm all are within `[0, 2**31 - 1]`. _Verified: 0 out-of-range seeds across 144 records; observed range 9_375_410 .. 2_102_105_591._

## 5. Documentation

- [x] 5.1 In `src/motco/simulations/study/README.md`, add a one-line note in the troubleshooting section explaining that the seed-derivation version bumped in this change invalidates pre-fix shards automatically — operators only need to re-run.
