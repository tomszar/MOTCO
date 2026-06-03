## Why

The semi-synthetic generator supports five `trajectory_mode` values (`none`, `translation`, `magnitude`, `orientation`, `shape`), each deforming a group trajectory along a different geometric axis, but there is no way to *see* what each mode does. An illustrative side-by-side visualization makes the geometry concrete for documentation, teaching, and sanity-checking the perturbation logic.

> Note: this change is documented **as-built** — the implementation already exists, is tested, and passes lint/mypy/pytest. The artifacts capture the change retroactively to keep the specs in sync.

## What Changes

- Add `src/motco/simulations/showcase.py` — an illustrative demo (distinct from the power study) that:
  - injects each requested `trajectory_mode` into **one shared InterSIM baseline** reusing the same seed, so stage and group assignment are identical across scenarios and only the injected effect differs (`none` is always zero-effect — the null)
  - concatenation-integrates each scenario into an outcome matrix
  - projects each scenario through **its own** 2-component PLS-DA with stage as the response and **no cross-validation**
  - renders a multi-panel trajectory figure
  - exposes `run_trajectory_showcase()` (entry point), `generate_showcase_datasets()`, `build_showcase_figure()`, `TRAJECTORY_SHOWCASE_MODES`, `TrajectoryShowcaseError`
- Add `stats/pls.py` `fit_plsda_model(X, y, n_components) -> PLSRegression` returning the fitted projector (no CV); `fit_plsda_transform` becomes a thin wrapper over it
- Add `viz.py` `plot_trajectory_from_plsr(...)` (PLS analog of `plot_trajectory_from_data`, returns the fitted PLS) and a `component_label` parameter on `plot_trajectories` so axes can read `PLS1/PLS2`
- Add `scripts/trajectory_showcase.py` — thin CLI that runs the showcase and saves the figure
- Export the new public names from `stats`, `simulations`, and the package root

## Capabilities

### New Capabilities

- `trajectory-mode-showcase`: shared-baseline generation of one dataset per `trajectory_mode` and a multi-panel per-scenario PLS-DA trajectory figure, plus the CLI that drives it

### Modified Capabilities

- `plsr-latent-space`: add `fit_plsda_model` (fitted-projector variant of `fit_plsda_transform`)
- `trajectory-geometry-plot`: add the PLS-DA wrapper `plot_trajectory_from_plsr` and the `component_label` axis-label parameter on `plot_trajectories`

## Impact

- New files: `src/motco/simulations/showcase.py`, `scripts/trajectory_showcase.py`, `tests/test_showcase.py`
- Modified: `src/motco/stats/pls.py`, `src/motco/viz.py`, `src/motco/__init__.py`, `src/motco/stats/__init__.py`, `src/motco/simulations/__init__.py`, `tests/test_pls.py`, `tests/test_viz.py`
- No new dependencies (`matplotlib`, `scikit-learn`, `rpy2` already present)
- Runtime constraint: InterSIM feature counts are fixed by its reference data (~650 features total across the three layers) and are not parameterizable; `n_sample` and the number of stages (clusters) are controllable
- The showcase requires R + InterSIM at runtime; tests stub the InterSIM result and run without R
