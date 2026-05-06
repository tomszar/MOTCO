## Requirements

### Requirement: Core plot function renders trajectory geometry from pre-computed vectors
MOTCO SHALL provide `plot_trajectories(observed_vectors, projector, ...)` that accepts a MultiIndex DataFrame of LS-mean vectors and a fitted projector, and returns a matplotlib `(Figure, Axes)` tuple with the trajectory geometry drawn.

#### Scenario: Two groups rendered as distinct paths
- **WHEN** `observed_vectors` contains LS-mean vectors for two groups across two or more stages
- **THEN** each group is drawn as a connected path of points in 2D projected space, using a distinct color per group

#### Scenario: Segment direction arrows indicate stage order
- **WHEN** a group trajectory has two or more stages
- **THEN** each connecting segment between consecutive LS-mean points has a directional arrow at its midpoint pointing toward the later stage

#### Scenario: First stage is labelled
- **WHEN** a group trajectory is rendered
- **THEN** the first stage point is annotated with the stage label to anchor the temporal order

#### Scenario: Axis labels include explained variance
- **WHEN** the projector exposes explained variance ratios (e.g., a fitted sklearn PCA)
- **THEN** axis labels read "PC1 (XX.X%)" and "PC2 (XX.X%)"

#### Scenario: Caller supplies existing axes
- **WHEN** an `ax` argument is provided
- **THEN** the plot is drawn onto that axes object and the same axes is returned in the tuple

#### Scenario: Sample scatter is hidden by default
- **WHEN** `show_samples` is not set or is `False`
- **THEN** no individual sample points are drawn

#### Scenario: Sample scatter is shown when opted in
- **WHEN** `show_samples=True` and `samples` and `sample_metadata` are provided
- **THEN** individual sample points are drawn as small semi-transparent markers behind the trajectory, colored by group

### Requirement: Wrapper function fits PCA and returns projector
MOTCO SHALL provide `plot_trajectory_from_data(Y, metadata, group_col, level_col, ...)` that fits a PCA on the outcome matrix `Y`, calls `plot_trajectories`, and returns `(Figure, Axes, PCA)`.

#### Scenario: PCA is fitted on all samples
- **WHEN** `plot_trajectory_from_data` is called with an outcome matrix `Y`
- **THEN** PCA is fitted on all rows of `Y` before projection

#### Scenario: Fitted PCA is returned for reuse
- **WHEN** `plot_trajectory_from_data` returns
- **THEN** the return value is a three-tuple `(fig, ax, pca)` where `pca` is the fitted PCA object

#### Scenario: Wrapper passes through keyword arguments
- **WHEN** keyword arguments such as `show_samples`, `palette`, or `ax` are supplied to the wrapper
- **THEN** they are forwarded to `plot_trajectories`

### Requirement: Both functions are exported from the package root
MOTCO SHALL export `plot_trajectories` and `plot_trajectory_from_data` from `src/motco/__init__.py` so callers can import them directly from `motco`.

#### Scenario: Top-level import works
- **WHEN** a caller writes `from motco import plot_trajectories`
- **THEN** the import succeeds without importing from a submodule path
