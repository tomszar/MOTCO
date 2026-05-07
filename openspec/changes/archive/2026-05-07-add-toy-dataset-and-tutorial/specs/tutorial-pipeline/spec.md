## ADDED Requirements

### Requirement: End-to-end notebook using toy data
`examples/motco_example.ipynb` SHALL be rewritten to demonstrate the complete MOTCO pipeline using the pre-generated toy dataset at `examples/data/toy/`. The notebook SHALL contain the following sections in order:

1. **Data generation** — shows the `motco simulate` command and notes the R + InterSIM dependency; loads the pre-generated data from `examples/data/toy/`.
2. **PLS-DA latent space** — demonstrates supervised dimension reduction: concatenates omics layers, runs `plsda_doubleCV` with `y = stage`, fits a final model, and produces a score matrix Y.
3. **SNF latent space** — demonstrates unsupervised integration as an alternative path to Y.
4. **Trajectory design** — loads pre-generated design files (`model_full.csv`, `model_reduced.csv`, `ls_means.csv`, `contrast.json`) from `examples/data/toy/`.
5. **Differential trajectory analysis** — runs `estimate_difference` and `RRPP` on Y from both integration paths; prints angle, delta, and p-values.
6. **Visualization** — uses `viz.py` (`plot_trajectory_from_data` or `plot_trajectories`) to render the group trajectories.

#### Scenario: Notebook runs to completion without errors
- **WHEN** the notebook is executed with `jupyter nbconvert --to notebook --execute`
- **THEN** all cells complete without exception and all output cells are populated

#### Scenario: PLS section uses stage as supervision signal
- **WHEN** the PLS section runs
- **THEN** the label passed to `plsda_doubleCV` is the `stage` column from `metadata.csv`, not the `group` column

#### Scenario: Both PLS and SNF paths produce a valid Y for de
- **WHEN** the notebook runs both integration sections
- **THEN** each produces a DataFrame with shape (n_samples, n_components) used as input to `estimate_difference`

### Requirement: README quick-start section
`README.md` SHALL include a "Quick start" section near the top that shows the complete 4-command CLI pipeline using the pre-generated toy data. The section SHALL note that `motco simulate` requires R + InterSIM and provide the install instruction for InterSIM.

#### Scenario: Quick start commands are copy-pasteable
- **WHEN** a user follows the quick start commands in sequence
- **THEN** all four commands (`motco simulate` or note about pre-generated data, `motco plsr`, `motco snf`, `motco de`) run without error using files in `examples/data/toy/`
