## Why

MOTCO has no browsable API reference — users must read source code to discover function signatures, parameters, and return types. Setting up MkDocs with mkdocstrings will auto-generate API docs from existing docstrings and type annotations, closing issue #13.

## What Changes

- Add `mkdocs`, `mkdocs-material`, and `mkdocstrings[python]` as optional `docs` dependencies in `pyproject.toml`
- Add `mkdocs.yml` at the project root configuring the Material theme, navigation, and mkdocstrings plugin
- Add a `docs/` directory with an index page and one API reference page per stats module (`pls`, `snf`, `sd`) plus the CLI
- Add a `docs` extras group so users can install docs dependencies with `uv sync --extra docs`
- Add a GitHub Actions workflow job to build and publish docs to GitHub Pages on push to `main`

## Capabilities

### New Capabilities

- `api-reference`: Auto-generated HTML API reference for all public functions in `motco.stats` and the `motco` CLI, built from docstrings and type annotations using mkdocstrings, served via MkDocs Material

### Modified Capabilities

*(none — no existing spec-level requirements are changing)*

## Impact

- **New files**: `mkdocs.yml`, `docs/index.md`, `docs/api/pls.md`, `docs/api/snf.md`, `docs/api/sd.md`, `docs/api/cli.md`, `.github/workflows/docs.yml`
- **Modified files**: `pyproject.toml` (new `docs` extras group)
- **Dependencies**: `mkdocs>=1.6`, `mkdocs-material>=9.5`, `mkdocstrings[python]>=0.25`
- **No breaking changes** to the Python API or CLI