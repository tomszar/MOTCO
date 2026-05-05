## Context

`RRPP` (in `sd.py`) and `plsda_doubleCV` (in `pls.py`) are the two long-running entry points. Both run for minutes at default settings (RRPP: 999 permutations; PLS-DA: 8×30 = 240 outer folds). Currently both are completely silent. `estimate_betas` has a three-tier fallback chain (Cholesky → direct solve → lstsq) that is also silent; the lstsq fallback signals a potentially rank-deficient design matrix, which is a numerical correctness concern. The CLI is the primary user interface; notebook users are assumed to be technically proficient.

## Goals / Non-Goals

**Goals:**
- Show a tqdm progress bar during the RRPP serial permutation loop and the plsda_doubleCV outer CV2 loop
- Emit `logging.WARNING` when `estimate_betas` falls back to direct solve or lstsq
- Emit `logging.INFO` for the RRPP parallel path (worker count, permutation count)
- Give CLI users a `--verbose` flag that routes log output to stderr at DEBUG level
- Allow library callers (notebooks) to suppress the bar via `progress=False`

**Non-Goals:**
- Progress bar for the RRPP parallel path (workers run in child processes; IPC cost outweighs benefit since parallel runs are already fast)
- Progress bar for the plsda_doubleCV inner CV1 loop (runs fast; outer loop is the bottleneck)
- Structured log output (JSON, file handlers) — stdlib default handler to stderr is sufficient
- `--no-progress` CLI flag (users can redirect stderr or pass `progress=False` directly)
- `rich` or any other TUI library

## Decisions

### tqdm over manual logging for progress
A `logger.info("permutation 500/999")` approach produces noisy log lines that are hard to read and not suppressible independently of other WARNING/INFO messages. `tqdm` renders a compact, overwriting progress bar on stderr, respects TTY detection (auto-disables in non-interactive contexts), and is already present in most scientific Python environments. The `disable=not progress` parameter provides a clean opt-out.

### `progress` parameter on stats functions, not just CLI
Placing the bar in the stats functions rather than wrapping them in the CLI keeps progress available to notebook users who call the API directly. Default `True` matches the CLI-primary design; passing `False` requires one explicit argument. The alternative (CLI-only wrapping) would leave library callers permanently silent.

### stdlib `logging` for warnings, not tqdm
Fallback events in `estimate_betas` are low-frequency, not loop-tick events — they happen at most once per `RRPP` or `estimate_difference` call. `logging.WARNING` is the right primitive: it's always visible (WARNING is the default level), integrates with any handler the caller configures, and is suppressible via the standard logging hierarchy. tqdm would be inappropriate here.

### `--verbose` sets root logger to DEBUG
The CLI has no persistent configuration file, so a simple flag is the right mechanism. `logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)` in `main()` when `--verbose` is passed covers all module loggers (`motco.stats.sd`, `motco.stats.pls`) without per-module configuration. Without `--verbose`, `basicConfig` is not called at all, leaving the root logger at WARNING (Python default), so library users who configure their own handlers are unaffected.

### tqdm as a runtime dependency
tqdm has no transitive dependencies and is ~50 KB. Adding it to `[project.dependencies]` (not just extras) is appropriate since the progress bar is part of the default CLI experience, not an optional feature.

## Risks / Trade-offs

- **tqdm in non-TTY environments** (CI, piped output): tqdm auto-detects non-TTY and disables itself, printing nothing. → No action needed; this is the correct behavior.
- **tqdm + multiprocessing**: The parallel RRPP path runs `pool.starmap` which blocks until all chunks complete. Wrapping it with tqdm would show 0% then 100% instantly, which is misleading. → Explicitly skip the bar for the parallel path; use `logger.info` instead.
- **New runtime dep**: Some downstream environments may pin tqdm or have conflicts. → tqdm is pinned loosely (`tqdm>=4.0`); version conflicts are extremely unlikely given its stability.

## Migration Plan

Additive change — no migration required. The `progress` parameter defaults to `True` (same visible behavior as adding a new feature). Callers who previously relied on silence can pass `progress=False`.
