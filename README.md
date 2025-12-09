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

Important: The `sd.py` utilities are now generalized and no longer assume dataset‑specific column names. See the Python API notes below for helpers to build model matrices and LS means directly from your `group` and `level` columns.

## Python API (selected)

```python
from motco.stats.pls import plsda_doubleCV
from motco.stats.snf import get_affinity_matrix, SNF, get_spectral
from motco.stats.sd import (
    estimate_difference,
    RRPP,
    center_matrix,
    get_model_matrix,
    build_ls_means,
    get_observed_vectors,
    pair_difference,
)

# Example: build model matrix and LS means from group/level columns
import pandas as pd
import numpy as np

# X_factors contains two categorical columns: 'group' and 'level'
X_factors = pd.DataFrame({
    'group': ['A','A','B','B'],
    'level': ['t0','t1','t0','t1'],
})

# Y is the outcome/feature matrix aligned by rows to X_factors (e.g., latent space)
Y = pd.DataFrame(np.random.randn(4, 3), columns=['z1','z2','z3'])

# Build design and estimate LS means for all group×level cells
X = get_model_matrix(X_factors, group_col='group', level_col='level', full=True)
ls = build_ls_means(
    group_levels=sorted(X_factors['group'].astype(str).unique()),
    level_levels=sorted(X_factors['level'].astype(str).unique()),
    full=True,
)
deltas, angles, shapes = estimate_difference(Y=Y.values, model_matrix=X, LS_means=ls, contrast=[[0,1],[2,3]])

# Two‑state comparison between two groups at two levels (angle & delta)
df = X_factors.copy()
df = pd.concat([df, Y], axis=1)
angle_deg, delta_mag = pair_difference(df, group_col='group', level_col='level', feature_cols=['z1','z2','z3'])

# Center features within groups (optional preprocessing)
df_centered = center_matrix(df, group_col='group', level_col='level', feature_cols=['z1','z2','z3'])
```

See inline docstrings in the modules under `src/motco/stats/` for full details.

### Breaking changes

- The statistical helpers in `motco.stats.sd` were generalized from dataset‑specific
  assumptions (e.g., columns like `PTGENDER` and `DX`) to explicit `group_col` and
  `level_col` parameters. There are no defaults; you must provide your column names.
  Legacy parameter names (e.g., `sex_col`) are removed.

## License

This project is licensed under the terms of the LICENSE file included in this repository.
