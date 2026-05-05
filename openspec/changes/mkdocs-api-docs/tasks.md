## 1. Add docs dependencies to pyproject.toml

- [x] 1.1 Add a `docs` optional-dependencies group to `pyproject.toml` with `mkdocs>=1.6`, `mkdocs-material>=9.5`, and `mkdocstrings[python]>=0.25`
- [x] 1.2 Run `uv sync --extra docs` and confirm the three packages install without conflicts

## 2. Create MkDocs configuration

- [x] 2.1 Create `mkdocs.yml` at the project root with: `site_name`, `repo_url` pointing to `https://github.com/tomszar/MOTCO`, Material theme, and mkdocstrings plugin configured with the Python handler
- [x] 2.2 Set `mkdocstrings` handler options: `show_source: true`, `show_root_heading: true`, `docstring_style: numpy` (confirmed NumPy style from existing docstrings)

## 3. Create docs source pages

- [x] 3.1 Create `docs/index.md` as the landing page — brief description of MOTCO, links to the three API sections, and a pointer to the README/example notebook
- [x] 3.2 Create `docs/api/pls.md` with `::: motco.stats.pls` mkdocstrings directive covering `plsda_doubleCV` and `calculate_vips`
- [x] 3.3 Create `docs/api/snf.md` with `::: motco.stats.snf` directive covering `get_affinity_matrix`, `SNF`, and `get_spectral`
- [x] 3.4 Create `docs/api/sd.md` with `::: motco.stats.sd` directive covering all public sd functions (`get_model_matrix`, `build_ls_means`, `estimate_difference`, `RRPP`, `estimate_betas`, `get_observed_vectors`, `pair_difference`, `center_matrix`)
- [x] 3.5 Create `docs/api/cli.md` documenting the three CLI subcommands (`plsr`, `snf`, `de`) — since the CLI is argparse-based (not a Python API), write this as a hand-authored Markdown page with the full `--help` output for each subcommand

## 4. Wire up navigation in mkdocs.yml

- [x] 4.1 Add a `nav:` section to `mkdocs.yml` with: Home → `index.md`, API Reference → PLS (`api/pls.md`), SNF (`api/snf.md`), SD (`api/sd.md`), CLI (`api/cli.md`)

## 5. Verify local build

- [x] 5.1 Run `mkdocs build --strict` and confirm exit code 0 with no warnings
- [ ] 5.2 Run `mkdocs serve` and manually check each API page renders correctly (signatures, types, docstrings visible)

## 6. Add GitHub Actions workflow for Pages deployment

- [x] 6.1 Create `.github/workflows/docs.yml` that triggers on push to `main`, installs `uv`, runs `uv sync --extra docs`, installs `motco` (so mkdocstrings can resolve imports), then runs `mkdocs gh-deploy --force`
- [x] 6.2 Ensure the workflow job has `permissions: contents: write` (required for `gh-deploy` to push to `gh-pages`)

## 7. Enable GitHub Pages and verify deployment

- [ ] 7.1 Push changes to `main` and confirm the `docs.yml` workflow completes successfully
- [ ] 7.2 In the GitHub repo Settings → Pages, set Source to `gh-pages` branch (root) if not already configured
- [ ] 7.3 Confirm the published site is reachable at `https://tomszar.github.io/MOTCO/`

## 8. Close issue and update README

- [x] 8.1 Add a "Documentation" badge or link to the published site in `README.md`
- [ ] 8.2 Close GitHub issue #13 referencing the relevant commits
