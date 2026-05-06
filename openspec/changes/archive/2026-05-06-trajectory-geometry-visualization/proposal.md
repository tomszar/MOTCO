## Why

MOTCO computes trajectory geometry (size, orientation, shape) and pairwise comparison statistics, but offers no way to visualize what those trajectories look like in the feature space. Without a visual, it is difficult to build intuition about the geometry or sanity-check results during exploration.

## What Changes

- Add `src/motco/viz.py` — a new top-level visualization module
- Expose a two-layer plotting API:
  - `plot_trajectories()` — core function; takes pre-computed LS-mean vectors and a fitted projector, draws the geometry
  - `plot_trajectory_from_data()` — convenience wrapper; fits PCA on the outcome matrix, calls the core function, returns the fitted projector for reuse
- Export both functions from `src/motco/__init__.py`

## Capabilities

### New Capabilities

- `trajectory-geometry-plot`: 2D PCA projection of LS-mean trajectory paths for two or more groups, with per-segment direction arrows, optional sample scatter, and PC-variance axis labels

### Modified Capabilities

<!-- none -->

## Impact

- New file: `src/motco/viz.py`
- `src/motco/__init__.py`: add exports for `plot_trajectories` and `plot_trajectory_from_data`
- New dependency: `matplotlib` (add to `pyproject.toml` if not already present)
- `scikit-learn` already a dependency (used for PCA)
- No changes to existing stats modules or CLI
