## 1. Dependencies and Module Setup

- [x] 1.1 Add `matplotlib` to runtime dependencies in `pyproject.toml`
- [x] 1.2 Create `src/motco/viz.py` with module docstring and imports

## 2. Core Plot Function

- [x] 2.1 Implement `plot_trajectories(observed_vectors, projector, ax, show_samples, samples, sample_metadata, group_col, level_col, palette)` — project LS-mean vectors through projector and draw one connected path per group
- [x] 2.2 Add per-segment direction arrows at midpoints using `matplotlib` annotation
- [x] 2.3 Annotate the first stage point of each group with the stage label
- [x] 2.4 Set axis labels from projector explained variance ratios when available (e.g., "PC1 (34.2%)")
- [x] 2.5 Implement optional sample scatter: project `samples` through projector and draw small semi-transparent points colored by group when `show_samples=True`

## 3. Wrapper Function

- [x] 3.1 Implement `plot_trajectory_from_data(Y, metadata, group_col, level_col, full, n_components, ax, show_samples, palette)` — call `get_observed_vectors`, fit PCA on `Y`, call `plot_trajectories`, return `(fig, ax, pca)`

## 4. Package Exports

- [x] 4.1 Export `plot_trajectories` and `plot_trajectory_from_data` from `src/motco/__init__.py`

## 5. Tests

- [x] 5.1 Write smoke test for `plot_trajectories` with synthetic 2-group × 2-stage LS-mean vectors and a fitted PCA — assert a Figure is returned and no exceptions raised
- [x] 5.2 Write smoke test for `plot_trajectory_from_data` with synthetic `Y` and metadata — assert `(fig, ax, pca)` returned and `pca` is a fitted sklearn PCA
- [x] 5.3 Test `show_samples=True` path — assert scatter artists are present in axes
- [x] 5.4 Test top-level import: `from motco import plot_trajectories, plot_trajectory_from_data`
