# Repository Guidelines

## Project Structure & Module Organization

MOTCO is a Python package using the `src/` layout. Core package code lives in `src/motco/`; the CLI entry point is `src/motco/cli.py`, and statistical routines are split under `src/motco/stats/` (`pls.py`, `snf.py`, `design.py`, `trajectory.py`, `permutation.py`). Tests live in `tests/`, with fixtures and reference CSV/R data under `tests/data/`. Documentation is in `docs/` and is built with MkDocs using `mkdocs.yml`. Example usage is in `examples/motco_example.ipynb`. OpenSpec proposals and canonical specs are under `openspec/`.

## Build, Test, and Development Commands

Use Python 3.11+ and `uv`.

```bash
uv venv
source .venv/bin/activate
uv sync --extra test --extra docs
uv run pytest tests/ -m "not slow" --tb=short
uv run ruff check src/ tests/
uv run mypy src/motco/
uv run mkdocs serve
```

`uv sync` installs locked dependencies and optional test/docs tooling. `pytest` runs the test suite; use `MOTCO_TEST_PERMS=99` for faster permutation tests during development. `ruff` checks imports and style. `mypy` type-checks package code. `mkdocs serve` previews documentation locally. The CLI can be inspected with `uv run motco --help`.

## Coding Style & Naming Conventions

Target Python 3.11. Keep line length at 120 characters, matching `pyproject.toml`. Ruff enforces `E`, `F`, `W`, and import-order (`I`) rules. Prefer explicit, descriptive function and variable names for statistical code; keep public APIs stable and documented with docstrings. Test files should follow `test_*.py` or `*_test.py`.

## Testing Guidelines

Tests use `pytest` with configuration in `pyproject.toml`. Slow tests are marked `slow`; run fast tests with:

```bash
MOTCO_TEST_PERMS=99 uv run pytest tests/ -m "not slow" --tb=short
```

For full regression coverage, use `MOTCO_TEST_PERMS=10000 MOTCO_N_JOBS=-1 uv run pytest tests/ --tb=short`. Add tests alongside behavior changes, especially for numerical fallbacks, CLI validation, and reference-data comparisons.

## Commit & Pull Request Guidelines

Follow the existing Conventional Commit style: `feat:`, `fix:`, `docs:`, `chore:`, and `refactor:`. Keep commits focused and imperative, for example `fix: handle singular model matrix fallback`. Pull requests should include a short problem statement, summary of changes, commands run, linked issues or OpenSpec changes when relevant, and screenshots only for documentation or rendered output changes.

## Agent-Specific Instructions

When running shell commands as an agent, prefix commands with `rtk` as configured by the local repository instructions. Do not overwrite user changes; inspect the worktree before broad edits and keep changes scoped to the requested task.
