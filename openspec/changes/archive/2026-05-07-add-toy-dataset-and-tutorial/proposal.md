## Why

MOTCO has no entry point for new users to experience the full pipeline. The existing notebook starts from pre-processed 2D data, skipping the omics integration steps that are central to the method. A toy dataset generated from the InterSIM simulation bridge — with a known orientation-mode trajectory difference injected — gives users a concrete, runnable example of the complete workflow: simulate → integrate (PLS or SNF) → differential trajectory analysis.

## What Changes

- **`cli.py`** — add `motco simulate` subcommand: wraps `generate_semisynthetic_trajectory_from_intersim` + `build_simulation_trajectory_design` to produce all files needed for a full pipeline run in one command. Fails gracefully if R / InterSIM is unavailable.
- **`examples/data/toy/`** — pre-generated toy dataset committed to the repo (seed 42, orientation mode, effect size 2.0, 90 samples). Users without R can run the tutorial immediately.
- **`examples/motco_example.ipynb`** — rewritten to show the complete pipeline using toy data: data generation → PLS-DA latent space → SNF latent space → trajectory design → `estimate_difference` + RRPP → visualization.
- **`README.md`** — new "Quick start" section at the top showing the 4-command CLI pipeline; notes InterSIM as an optional R dependency for data generation.

## Capabilities

### New Capabilities

- `simulate-command`: CLI subcommand `motco simulate` that generates aligned multi-omics matrices (methylation, expression, proteomics), sample metadata, and a complete trajectory design (model matrices, LS means, contrast) in a single output directory. Parametrized by seed, sample count, trajectory mode, and effect size.
- `toy-dataset`: Pre-generated example dataset bundled with the repository at `examples/data/toy/`. Demonstrates an orientation-mode group trajectory difference across three disease stages. Usable without R.
- `tutorial-pipeline`: End-to-end notebook and README quick start that walks users through the full MOTCO workflow using the toy dataset, covering both the PLS-DA and SNF integration paths.

### Modified Capabilities

(none)

## Impact

- `src/motco/cli.py`: new `cmd_simulate` function and `simulate` subparser entry.
- `examples/data/toy/`: new directory with ~10 committed CSV/JSON files.
- `examples/motco_example.ipynb`: rewritten (existing file replaced).
- `README.md`: new section added at top.
- `tests/`: tests for `cmd_simulate` (happy path + missing R error).
- Requires `add-plsr-latent-space` merged first (tutorial uses `--out-scores`).
- Optional runtime dependency: R + InterSIM package (for `motco simulate`; not required for tutorial usage of pre-generated data).
