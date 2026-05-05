## 1. Add tqdm runtime dependency

- [x] 1.1 Add `tqdm>=4.0` to `[project.dependencies]` in `pyproject.toml`
- [x] 1.2 Run `uv sync` and confirm tqdm installs without conflicts

## 2. Add logging warnings to estimate_betas in sd.py

- [x] 2.1 Add `import logging` and `logger = logging.getLogger(__name__)` near the top of `sd.py`
- [x] 2.2 In `estimate_betas`, add `logger.warning(...)` in the `except` block that falls back to direct solve (after the first `LinAlgError`)
- [x] 2.3 In `estimate_betas`, add `logger.warning(...)` in the `except` block that falls back to `lstsq` (after the second `LinAlgError`)
- [x] 2.4 Run `uv run pytest tests/test_validation.py tests/test_sd_smoke.py -v` to confirm no regressions

## 3. Add tqdm progress bar to RRPP serial path in sd.py

- [x] 3.1 Add `from tqdm import tqdm` import to `sd.py`
- [x] 3.2 Add `progress: bool = True` parameter to the `RRPP` function signature
- [x] 3.3 Wrap the serial `for _ in range(permutations):` loop: `for _ in tqdm(range(permutations), desc="RRPP", unit="perm", disable=not progress):`
- [x] 3.4 Add `logger.info("Running %d permutations across %d workers", permutations, n_workers)` at the start of the parallel path (before `pool.starmap`)
- [x] 3.5 Run `uv run pytest tests/test_validation.py -v` to confirm RRPP tests still pass

## 4. Add tqdm progress bar to plsda_doubleCV outer loop in pls.py

- [x] 4.1 Add `import logging`, `logger = logging.getLogger(__name__)`, and `from tqdm import tqdm` near the top of `pls.py`
- [x] 4.2 Add `progress: bool = True` parameter to the `plsda_doubleCV` function signature
- [x] 4.3 Replace `for rest, test in cv2.split(X, y):` with tqdm-wrapped iterator (extracted to `cv2_iter` to respect line length)
- [x] 4.4 Run `uv run pytest tests/test_pls.py -v` to confirm PLS tests still pass

## 5. Add --verbose flag to CLI

- [x] 5.1 Add `p.add_argument("--verbose", action="store_true", help="Enable debug logging to stderr")` to `build_parser()` in `cli.py`, before the subparsers
- [x] 5.2 In `main()`, add: `if args.verbose: logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)` before `args.func(args)` — also add `import logging` and `import sys` to `cli.py` if not already present
- [x] 5.3 Run `uv run pytest tests/test_cli.py -v` to confirm CLI tests still pass

## 6. Full test suite and lint

- [x] 6.1 Run `MOTCO_TEST_PERMS=99 uv run pytest tests/ -m "not slow" -v` and confirm all tests pass
- [x] 6.2 Run `uv run ruff check src/ tests/` and fix any issues
- [x] 6.3 Run `uv run mypy src/motco/` and fix any type errors (added tqdm.* to mypy overrides in pyproject.toml)

## 7. Commit

- [x] 7.1 Commit all changes
- [x] 7.2 Push and close issue #4
