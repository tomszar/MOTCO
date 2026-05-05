# CLI Reference

The `motco` command-line interface provides three subcommands.

```
usage: motco [-h] [--version] {plsr,snf,de} ...

MOTCO CLI: PLSR, SNF, and group differences

positional arguments:
  {plsr,snf,de}
    plsr         Run PLS-DA with double cross-validation
    snf          Similarity Network Fusion
    de           Group differences on trajectories (estimate or RRPP)
```

---

## `motco plsr`

Run PLS-DA with double cross-validation. Input can be a single CSV (with `--data` and `--label-col`) or separate predictor/label CSVs (`--x` / `--y`).

```
usage: motco plsr [-h] [--data DATA] [--label-col LABEL_COL] [--x X] [--y Y]
                  [--cv1-splits CV1_SPLITS] [--cv2-splits CV2_SPLITS]
                  [--n-repeats N_REPEATS] [--max-components MAX_COMPONENTS]
                  [--random-state RANDOM_STATE] [--out-table OUT_TABLE]
                  [--out-vips OUT_VIPS]

options:
  --data DATA                CSV with predictors and label column
  --label-col LABEL_COL      Label column name when using --data
  --x X                      CSV with predictors (features)
  --y Y                      CSV with labels/outcomes
  --cv1-splits CV1_SPLITS    Inner CV folds
  --cv2-splits CV2_SPLITS    Outer CV folds
  --n-repeats N_REPEATS      Number of outer CV repeats
  --max-components MAX_COMPONENTS
                             Maximum latent variables to search
  --random-state RANDOM_STATE
                             Random seed for reproducibility
  --out-table OUT_TABLE      Path to save the best models table (CSV)
  --out-vips OUT_VIPS        Path to save VIP scores per feature (CSV)
```

**Example:**

```bash
motco plsr \
  --data data.csv --label-col group \
  --cv1-splits 5 --cv2-splits 5 --n-repeats 10 --max-components 5 \
  --out-table results/plsr_table.csv \
  --out-vips results/vips.csv
```

---

## `motco snf`

Similarity Network Fusion across two or more omics CSV files. Pass `--input` once per dataset (all must have the same number of rows in the same sample order).

```
usage: motco snf [-h] [--input INPUT] [--K K] [--eps EPS] [--k K] [--t T]
                 [--out-fused OUT_FUSED] [--out-embedding OUT_EMBEDDING]
                 [--spectral-components SPECTRAL_COMPONENTS]

options:
  --input INPUT              Input CSV (repeat for multiple omics)
  --K K                      K for affinity construction (default: 20)
  --eps EPS                  Epsilon for affinity construction (default: 0.5)
  --k K                      k for sparse kernel in SNF (default: 20)
  --t T                      Number of SNF iterations (default: 20)
  --out-fused OUT_FUSED      Path to save fused matrix (CSV)
  --out-embedding OUT_EMBEDDING
                             Path to save spectral embedding (CSV)
  --spectral-components SPECTRAL_COMPONENTS
                             Number of spectral embedding components (default: 10)
```

**Example:**

```bash
motco snf \
  --input proteomics.csv --input metabolomics.csv \
  --K 20 --k 20 --t 20 \
  --out-fused results/fused.csv \
  --out-embedding results/embedding.csv \
  --spectral-components 10
```

---

## `motco de`

Estimate group trajectory differences, optionally with RRPP permutation testing.

**Estimate only** (no p-values): provide `--model-matrix`.

**RRPP** (with p-values): provide `--model-full`, `--model-reduced`, and `--rrpp-permutations`.

```
usage: motco de [-h] --Y Y --ls-means LS_MEANS --contrast CONTRAST
                [--model-matrix MODEL_MATRIX] [--model-full MODEL_FULL]
                [--model-reduced MODEL_REDUCED]
                [--rrpp-permutations RRPP_PERMUTATIONS] [--out-json OUT_JSON]
                [--out-observed OUT_OBSERVED]

options:
  --Y Y                      Outcome matrix CSV (latent space coordinates)
  --ls-means LS_MEANS        Least-squares means CSV
  --contrast CONTRAST        JSON file with groups (list of index lists)
  --model-matrix MODEL_MATRIX
                             Model matrix CSV (with intercept) for estimate_difference
  --model-full MODEL_FULL    Full model matrix CSV (with intercept) for RRPP
  --model-reduced MODEL_REDUCED
                             Reduced model matrix CSV (with intercept) for RRPP
  --rrpp-permutations RRPP_PERMUTATIONS
                             Number of permutations for RRPP
  --out-json OUT_JSON        Output JSON file
  --out-observed OUT_OBSERVED
                             Save predicted LS-mean vectors as CSV
```

**Example — estimate only:**

```bash
motco de \
  --Y latent.csv --model-matrix model_full.csv \
  --ls-means ls_means.csv --contrast contrast.json \
  --out-json results/de.json \
  --out-observed results/ls_mean_vectors.csv
```

**Example — RRPP:**

```bash
motco de \
  --Y latent.csv \
  --model-full model_full.csv --model-reduced model_reduced.csv \
  --ls-means ls_means.csv --contrast contrast.json \
  --rrpp-permutations 999 \
  --out-json results/rrpp.json
```
