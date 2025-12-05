# MOTCO — Multi-omics Trajectory Comparison

MOTCO provides tooling to generate latent spaces from multi‑omics data and to test for group differences in multivariate trajectories within those spaces.

Two approaches are implemented to build latent spaces:

- Partial Least Squares Regression/Discriminant Analysis (PLSR/PLS‑DA)
- Similarity Network Fusion (SNF) with optional spectral embedding

Once a latent space is constructed, MOTCO includes statistics to estimate differences in multivariate trajectories between groups (magnitude, orientation/angle, and shape), with an option for permutation testing via RRPP (Residual Randomization in a Permutation Procedure).

This repository contains the core statistical routines in `src/motco/stats` and a command‑line interface for common tasks.

## Install (with uv)

Prerequisites: Python 3.9+ and [uv](https://github.com/astral-sh/uv) installed.

```bash
# Create and activate a virtual environment
uv venv
source .venv/bin/activate

# Install MOTCO in editable mode (and dependencies)
uv pip install -e .

# Verify CLI is available
motco --help
```

Alternatively, using pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Command Line Interface

MOTCO exposes a single entry‑point `motco` with subcommands for PLSR/PLS‑DA, SNF, and Differential Effects (group differences).

### 1) PLS‑DA with double cross‑validation

```bash
motco plsr \
  --data path/to/table.csv \
  --label-col diagnosis \
  --cv1-splits 7 --cv2-splits 8 --n-repeats 30 --max-components 50 \
  --out-table results/plsr_models.csv
```

Options:
- Use `--data` for a single CSV containing predictors and a label column specified by `--label-col`.
- Or provide separate matrices via `--x` and `--y` CSV files (mutually exclusive with `--data`).
- Outputs a table with the best model per outer CV repeat (LV and AUROC). The actual trained models are kept in memory; export of models is not included in the CLI at this time.

Input expectations:
- CSV files with samples in rows, features in columns. For `--data`, include a label column with binary or multi‑class outcome; it will be one‑hot encoded internally.

### 2) Similarity Network Fusion (SNF)

```bash
motco snf \
  --input omics1.csv --input omics2.csv [--input omics3.csv ...] \
  --K 20 --eps 0.5 --k 20 --t 20 \
  --out-fused fused_affinity.csv \
  --out-embedding spectral_embedding.csv
```

Notes:
- Each `--input` CSV must contain the same samples in the same order (rows = samples).
- `--K` and `--eps` are used when constructing per‑dataset affinity matrices; `--k` and `--t` control SNF neighborhood size and iterations.
- The fused similarity matrix is saved to `--out-fused`. If `--out-embedding` is provided, a 10‑dimensional spectral embedding is also computed and saved.

### 3) Differential Effects (group differences)

Estimate differences in magnitude and direction between groups using least‑squares means, with optional permutation testing (RRPP).

```bash
motco de \
  --Y latent_space.csv \
  --model-matrix model_matrix.csv \
  --ls-means ls_means.csv \
  --contrast contrast.json \
  --out-json de_result.json

# With permutations (RRPP)
motco de \
  --Y latent_space.csv \
  --model-full model_full.csv \
  --model-reduced model_reduced.csv \
  --ls-means ls_means.csv \
  --contrast contrast.json \
  --rrpp-permutations 999 \
  --out-json rrpp_result.json
```

Where:
- `latent_space.csv` is the outcome matrix `Y` (e.g., coordinates in a latent space; rows = samples, columns = dimensions).
- `model_matrix.csv` is a design matrix (with intercept) aligned to `Y` rows. For RRPP, provide `--model-full` and `--model-reduced` (both with intercept).
- `ls_means.csv` contains the least‑squares means to compare (rows = groups/cohorts, columns = same dimensions as `Y`).
- `contrast.json` is a JSON array of index lists, where each inner list enumerates the cohort indices belonging to the same group. Example: `[[0,1],[2,3]]`.
- Output JSON will include delta (magnitude) and angle (direction) matrices; RRPP adds distributions per permutation.

Important: The `sd.py` utilities include helpers tailored to a specific example dataset (e.g., `get_model_matrix` expects certain categorical columns). For general workflows, prefer preparing explicit model matrices and LS means externally and passing them to the CLI as shown above.

## Python API (selected)

```python
from motco.stats.pls import plsda_doubleCV
from motco.stats.snf import get_affinity_matrix, SNF, get_spectral
from motco.stats.sd import estimate_difference, RRPP
```

See inline docstrings in the modules under `src/motco/stats/` for full details.

## License

This project is licensed under the terms of the LICENSE file included in this repository.
