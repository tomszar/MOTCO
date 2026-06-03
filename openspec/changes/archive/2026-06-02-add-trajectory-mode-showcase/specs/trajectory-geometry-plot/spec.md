## ADDED Requirements

### Requirement: Configurable projected-axis label prefix
`plot_trajectories` SHALL accept a `component_label` parameter (default `"PC"`) that sets the prefix of the projected-axis labels, so non-PCA projectors can be labelled appropriately.

#### Scenario: Default prefix is PC
- **WHEN** `plot_trajectories` is called without `component_label`
- **THEN** the axes are labelled with the `PC` prefix (preserving existing PCA behavior, including variance percentages when available)

#### Scenario: Custom prefix is applied
- **WHEN** `plot_trajectories` is called with `component_label="PLS"`
- **THEN** the axis labels read `PLS1` and `PLS2`

### Requirement: PLS-DA wrapper fits a supervised projector and returns it
MOTCO SHALL provide `plot_trajectory_from_plsr(Y, metadata, group_col, level_col, ...)` that fits a 2-component PLS-DA on `Y` with the level/stage factor as the response (no cross-validation), calls `plot_trajectories` with `component_label="PLS"`, and returns `(Figure, Axes, PLSRegression)`.

#### Scenario: PLS-DA is fitted on stage as the response
- **WHEN** `plot_trajectory_from_plsr` is called with outcome matrix `Y` and `level_col`
- **THEN** a `PLSRegression` is fitted on all rows of `Y` using the one-hot-encoded `level_col` values as the response

#### Scenario: Fitted PLS is returned for reuse
- **WHEN** `plot_trajectory_from_plsr` returns
- **THEN** the return value is a three-tuple `(fig, ax, pls)` where `pls` is the fitted `PLSRegression`

#### Scenario: Axes are labelled with the PLS prefix
- **WHEN** the figure is produced
- **THEN** the axis labels read `PLS1` and `PLS2`

#### Scenario: Wrapper forwards keyword arguments
- **WHEN** keyword arguments such as `show_samples`, `palette`, `ax`, or `n_components` are supplied
- **THEN** they are applied to the fit and/or forwarded to `plot_trajectories`

### Requirement: PLS-DA wrapper is exported from the package root
MOTCO SHALL export `plot_trajectory_from_plsr` from `src/motco/__init__.py` so callers can import it directly from `motco`.

#### Scenario: Top-level import works
- **WHEN** a caller writes `from motco import plot_trajectory_from_plsr`
- **THEN** the import succeeds without importing from a submodule path
