## Context

MOTCO is a Python package (`src/motco/`) with a public API exported from `motco.stats` (`__all__` lists 13 symbols across `pls.py`, `snf.py`, `sd.py`) and a CLI in `motco/cli.py`. The package uses `hatchling` for builds and `uv` for dependency management (`pyproject.toml` extras pattern already established with `test`). There are no docs infrastructure files (`docs/`, `mkdocs.yml`) today. The GitHub repo is at `github.com/tomszar/MOTCO`, which supports GitHub Pages.

## Goals / Non-Goals

**Goals:**
- Auto-generate browsable HTML API reference from existing docstrings and type annotations
- Support local preview with `mkdocs serve` (live-reload)
- Publish docs to GitHub Pages automatically on push to `main`
- Keep docs deps isolated under a `docs` extras group so they don't affect CI test installs

**Non-Goals:**
- Writing or improving existing docstrings (docs build from what exists)
- Narrative tutorials or how-to guides beyond the existing README/notebook
- Versioned documentation (single `latest` version only)
- Testing docstring content or accuracy

## Decisions

### MkDocs over Sphinx
Sphinx is the traditional Python docs tool but requires RST authorship and has complex configuration. MkDocs uses Markdown natively (consistent with this repo's existing docs), builds faster, and the Material theme provides search, mobile-friendliness, and a polished look out of the box. The tradeoff: Sphinx has deeper Python ecosystem integration (intersphinx, autodoc), but mkdocstrings now covers the core use case with less ceremony.

### mkdocstrings with Python handler
`mkdocstrings[python]` uses `griffe` to parse source ASTs and docstrings without importing the module, which avoids environment pollution during docs builds. It supports Google, NumPy, and reStructuredText docstring styles — all common in scientific Python. Signatures and type annotations are pulled from source even when docstrings are absent.

### One page per module
Rather than a single flat `api.md`, each stats module (`pls`, `snf`, `sd`) gets its own page, with a fourth page for the CLI. This mirrors the existing module structure and makes navigation predictable as the codebase grows.

### GitHub Pages via Actions, not `mkdocs gh-deploy`
`mkdocs gh-deploy` requires local credentials and is harder to automate reliably. The `peaceiris/actions-gh-pages` action (or the MkDocs Material built-in `gh-deploy` in CI) is the standard pattern for GitHub Pages + Actions. We'll use the `mkdocs gh-deploy --force` approach inside a workflow, keeping it simple without adding new Actions dependencies.

### `docs` extras group in pyproject.toml
Follows the existing `test` extras pattern. Docs deps are never needed at runtime or in CI test runs, so isolating them avoids version conflicts and keeps test installs lean.

## Risks / Trade-offs

- **Sparse docstrings**: Several internal helpers in `sd.py` have minimal or no docstrings. mkdocstrings will show the signature but no prose. → Mitigation: acceptable for now; the spec only requires public API symbols to be shown, not fully documented.
- **GitHub Pages branch conflict**: `gh-pages` branch must be initialized. → Mitigation: the workflow creates it automatically on first run.
- **`motco` must be importable during mkdocstrings build**: griffe avoids imports for most cases, but some dynamic attributes may require it. → Mitigation: install `motco` in the docs workflow step before building.

## Migration Plan

1. Add `docs` extras to `pyproject.toml`
2. Create `mkdocs.yml` and `docs/` structure
3. Test locally: `uv sync --extra docs && mkdocs serve`
4. Add `.github/workflows/docs.yml`
5. Push to `main` — first deploy creates the `gh-pages` branch and publishes
6. Enable GitHub Pages (Source: `gh-pages` branch) in repo Settings if not already set

No rollback needed — docs are additive and don't affect the package or tests.

## Open Questions

*(none — design is straightforward)*