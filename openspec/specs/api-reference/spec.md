## ADDED Requirements

### Requirement: All public API symbols are discoverable in the reference
The docs site SHALL render a dedicated API reference page for each of the four modules: `pls`, `snf`, `sd`, and `cli`. Each page SHALL display every symbol listed in `motco.stats.__all__` (for stats modules) and every public subcommand (for the CLI), including function signature, parameter names and types, return type, and docstring prose where present.

#### Scenario: User views the PLS module page
- **WHEN** a user navigates to the API Reference → PLS section of the docs site
- **THEN** they see `plsda_doubleCV` and `calculate_vips` with full signatures and type annotations

#### Scenario: User views a function with no docstring
- **WHEN** a user navigates to a function that has no prose docstring
- **THEN** the page still renders the function signature and parameter types without error

### Requirement: Docs site is buildable locally without errors
The docs site SHALL build to completion with `mkdocs build --strict` using only dependencies declared in the `docs` extras group, with zero warnings treated as errors.

#### Scenario: Developer runs local build
- **WHEN** a developer runs `uv sync --extra docs && mkdocs build --strict` in a clean environment
- **THEN** the build completes with exit code 0 and produces a `site/` directory

#### Scenario: Developer previews with live reload
- **WHEN** a developer runs `mkdocs serve`
- **THEN** a local server starts at `http://127.0.0.1:8000` with live-reload on file changes

### Requirement: Docs are published to GitHub Pages on push to main
The GitHub Actions workflow SHALL build and deploy the docs site to the `gh-pages` branch automatically whenever a commit is pushed to `main`.

#### Scenario: Commit pushed to main triggers deployment
- **WHEN** a commit is pushed to the `main` branch
- **THEN** the `docs.yml` workflow runs, builds the site, and force-pushes to `gh-pages` within 3 minutes

#### Scenario: Docs workflow does not run on feature branches
- **WHEN** a commit is pushed to any branch other than `main`
- **THEN** the `docs.yml` workflow does NOT trigger

### Requirement: Docs dependencies are isolated from test and runtime installs
The packages required to build docs (`mkdocs`, `mkdocs-material`, `mkdocstrings[python]`) SHALL be declared exclusively under a `docs` extras group in `pyproject.toml` and SHALL NOT be installed when running `uv sync --extra test`.

#### Scenario: CI test install does not pull in docs packages
- **WHEN** `uv sync --extra test` is run
- **THEN** `mkdocs`, `mkdocs-material`, and `mkdocstrings` are not present in the environment
