## ADDED Requirements

### Requirement: Shared-baseline generation of one dataset per trajectory mode
MOTCO SHALL provide `generate_showcase_datasets(intersim_result, *, modes, effect_size, seed, ...)` that injects each requested `trajectory_mode` into the **same** InterSIM baseline using the **same** seed, and returns an ordered mapping of mode name to `SemiSyntheticTrajectoryDataset`.

#### Scenario: Every requested mode is produced in order
- **WHEN** called with `modes=("none", "translation", "magnitude", "orientation", "shape")`
- **THEN** the returned mapping has exactly those keys in that order, each value a generated dataset

#### Scenario: none mode is always the null
- **WHEN** any `effect_size` is supplied
- **THEN** the dataset for `none` is generated with `group_effect_size == 0.0`

#### Scenario: Stage and group assignment is shared across modes
- **WHEN** datasets for two different modes are generated from one call
- **THEN** their per-sample `stage` and `group` assignments are identical (only the injected effect differs)

#### Scenario: Empty modes is rejected
- **WHEN** called with an empty `modes` collection
- **THEN** a `TrajectoryShowcaseError` is raised

### Requirement: Multi-panel per-scenario PLS-DA trajectory figure
MOTCO SHALL provide `build_showcase_figure(datasets, *, group_col, stage_col, ...)` that concatenation-integrates each dataset, projects each through its own 2-component PLS-DA (stage as response, no cross-validation), draws one trajectory panel per scenario, and returns a matplotlib `Figure`.

#### Scenario: One visible panel per scenario
- **WHEN** called with a mapping of N datasets
- **THEN** the returned figure has exactly N visible axes, one per scenario, and any extra grid cells are hidden

#### Scenario: Panels use the PLS projector
- **WHEN** a panel is rendered
- **THEN** its trajectory is drawn in the 2-component PLS-DA space and its axes are labelled with the `PLS` prefix

#### Scenario: Empty datasets is rejected
- **WHEN** called with an empty mapping
- **THEN** a `TrajectoryShowcaseError` is raised

### Requirement: End-to-end showcase entry point
MOTCO SHALL provide `run_trajectory_showcase(*, seed, n_sample, effect_size, modes, ...)` that runs InterSIM once, injects every mode into that shared baseline, builds the figure, and returns `(Figure, datasets)`.

#### Scenario: Returns figure and per-scenario datasets
- **WHEN** `run_trajectory_showcase` completes
- **THEN** it returns a two-tuple of the comparison `Figure` and the mode-to-dataset mapping

#### Scenario: Sample size is applied to InterSIM
- **WHEN** `n_sample` is provided (with or without explicit `intersim_params`)
- **THEN** InterSIM is invoked with that sample count

### Requirement: Showcase is driven by a thin CLI script
MOTCO SHALL provide `scripts/trajectory_showcase.py` that exposes `run_trajectory_showcase` from the command line and writes the figure to a file.

#### Scenario: Figure is written to the requested path
- **WHEN** the script is invoked with `--out <path>` and R + InterSIM are available
- **THEN** a figure file is written at `<path>` covering the requested modes

#### Scenario: Mode and sampling options are configurable
- **WHEN** `--modes`, `--n-sample`, `--effect-size`, or `--no-samples` are supplied
- **THEN** the rendered showcase reflects those options
