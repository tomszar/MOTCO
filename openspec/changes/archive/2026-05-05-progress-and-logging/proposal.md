## Why

Long-running operations (`RRPP` with thousands of permutations, `plsda_doubleCV` with tens of repeats) give no feedback, leaving CLI users staring at a blank terminal for minutes with no way to distinguish a slow run from a hung one. Numerical fallbacks in `estimate_betas` (Cholesky → direct solve → lstsq) are also completely silent, hiding potential rank-deficiency issues.

## What Changes

- Add `tqdm` progress bar to the `RRPP` serial permutation loop
- Add `tqdm` progress bar to the `plsda_doubleCV` outer CV2 loop
- Add `progress: bool = True` parameter to both functions so notebook users can opt out
- Add `logging.WARNING` messages when `estimate_betas` falls back to direct solve or lstsq
- Add `logging.INFO` message in `RRPP` parallel path (no bar possible; just announce worker count)
- Add `--verbose` flag to the `motco` CLI top-level parser; sets log level to `DEBUG`
- Add `tqdm` as a runtime dependency in `pyproject.toml`

## Capabilities

### New Capabilities

- `rrpp-progress`: Progress feedback and structured logging for RRPP permutation runs and plsda_doubleCV cross-validation, plus a CLI `--verbose` flag

### Modified Capabilities

*(none — no existing spec-level requirements are changing)*

## Impact

- **Modified files**: `src/motco/stats/sd.py`, `src/motco/stats/pls.py`, `src/motco/cli.py`, `pyproject.toml`
- **New runtime dependency**: `tqdm` (no transitive dependencies; ~50 KB)
- **API change**: `RRPP` and `plsda_doubleCV` gain a `progress` keyword argument (default `True`, backward-compatible)
- **No breaking changes**
