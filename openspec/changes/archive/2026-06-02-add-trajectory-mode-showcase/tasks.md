> Document-as-built: the implementation already exists and passes ruff + mypy + pytest. Tasks are recorded as complete to reflect the delivered state.

## 1. PLS projector helper

- [x] 1.1 Add `fit_plsda_model(X, y, n_components) -> PLSRegression` to `stats/pls.py`
- [x] 1.2 Refactor `fit_plsda_transform` to wrap `fit_plsda_model(...).x_scores_`
- [x] 1.3 Export `fit_plsda_model` from `stats/__init__.py`

## 2. PLS-DA trajectory plotting

- [x] 2.1 Add `component_label` parameter to `plot_trajectories` (default `"PC"`) and use it for axis labels
- [x] 2.2 Add `plot_trajectory_from_plsr(...)` to `viz.py` (fit 2-comp PLS-DA on stage, label `"PLS"`, return fitted PLS)
- [x] 2.3 Export `plot_trajectory_from_plsr` from `src/motco/__init__.py`

## 3. Showcase orchestration

- [x] 3.1 Add `simulations/showcase.py` with `TRAJECTORY_SHOWCASE_MODES`, `TrajectoryShowcaseError`, `_MODE_TITLES`
- [x] 3.2 Implement `generate_showcase_datasets` (shared baseline + shared seed; `none` forced to zero effect)
- [x] 3.3 Implement `build_showcase_figure` (concat-integrate, per-scenario PLS, grid layout, hide unused axes)
- [x] 3.4 Implement `run_trajectory_showcase` end-to-end entry point
- [x] 3.5 Export the showcase public names from `simulations/__init__.py`

## 4. CLI script

- [x] 4.1 Add `scripts/trajectory_showcase.py` (Agg backend, argparse, save figure)

## 5. Tests & gates

- [x] 5.1 Add `tests/test_showcase.py` (stub `InterSIMResult`, runs without R)
- [x] 5.2 Add `fit_plsda_model` coverage to `tests/test_pls.py`
- [x] 5.3 Add `plot_trajectory_from_plsr` coverage to `tests/test_viz.py`
- [x] 5.4 Confirm ruff + mypy + fast pytest all pass

## 6. Docs

- [x] 6.1 Note the `showcase.py` module in `CLAUDE.md`
