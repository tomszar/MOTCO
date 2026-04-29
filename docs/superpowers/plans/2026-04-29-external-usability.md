# MOTCO External Usability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make MOTCO usable by external scientists by adding input validation, test coverage, documentation, and minor API extensions.

**Architecture:** Four ordered layers — trust (validation guards in each stats module), coverage (new test files per module), accessibility (README, example notebook, CLI quality-of-life), extensions (expose VIPs, configurable spectral, clean `stats/__init__.py`). Complete each layer before starting the next.

**Tech Stack:** Python 3.11+, numpy, pandas, scikit-learn, scipy, pytest, uv

---

## File Map

| File | Action | Layer |
|------|--------|-------|
| `src/motco/stats/sd.py` | Add validation to `estimate_difference`, `RRPP`, `get_model_matrix`, `center_matrix` | 1 |
| `src/motco/stats/pls.py` | Add validation to `plsda_doubleCV`; rename `_calculate_vips` → `calculate_vips` | 1, 4 |
| `src/motco/stats/snf.py` | Add validation to `get_affinity_matrix`, `SNF`; add `n_components` to `get_spectral` | 1, 4 |
| `src/motco/cli.py` | Add `--version`, `--out-observed`, `--out-vips`, `--spectral-components` | 3, 4 |
| `src/motco/stats/__init__.py` | Export all public functions | 4 |
| `tests/test_validation.py` | New — one test per validation rule | 2 |
| `tests/test_pls.py` | New — PLS smoke tests | 2 |
| `tests/test_snf.py` | New — SNF smoke tests | 2 |
| `tests/test_cli.py` | New — CLI subcommand tests | 2 |
| `tests/test_sd_expected_example1.py` | Add `@pytest.mark.slow` | 2 |
| `tests/test_sd_expected_example2.py` | Add `@pytest.mark.slow` | 2 |
| `pyproject.toml` | Register `slow` marker | 2 |
| `README.md` | Add results interpretation, quick example | 3 |
| `examples/motco_example.ipynb` | New — end-to-end notebook | 3 |

---

## Task 1: Validate `estimate_difference` and `RRPP` inputs in `sd.py`

**Files:**
- Modify: `src/motco/stats/sd.py` (lines 250 and 325)
- Create: `tests/test_validation.py`

- [ ] **Step 1: Create `tests/test_validation.py` with failing tests**

```python
# tests/test_validation.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from motco.stats.sd import estimate_difference, RRPP, build_ls_means


def _make_simple_inputs(n_samples=10, n_features=3, n_groups=2, n_levels=2):
    """Return (Y, model_matrix, LS_means, contrast) for a 2-group 2-level design."""
    rng = np.random.default_rng(0)
    Y = rng.standard_normal((n_samples, n_features))
    # intercept + 1 group dummy + 1 level dummy + 1 interaction = 4 cols
    X = np.ones((n_samples, 4))
    X[:5, 1] = 0; X[5:, 1] = 1      # group dummy
    X[::2, 2] = 0; X[1::2, 2] = 1   # level dummy
    X[:, 3] = X[:, 1] * X[:, 2]     # interaction
    LS = build_ls_means(["A", "B"], ["t0", "t1"], full=True)  # shape (4, 4)
    contrast = [[0, 1], [2, 3]]
    return Y, X, LS, contrast


# --- estimate_difference: row mismatch ---
def test_estimate_difference_row_mismatch():
    Y, X, LS, contrast = _make_simple_inputs()
    X_bad = X[:8]  # 8 rows, Y has 10
    with pytest.raises(ValueError, match="10 rows"):
        estimate_difference(Y, X_bad, LS, contrast)


# --- estimate_difference: column mismatch ---
def test_estimate_difference_column_mismatch():
    Y, X, LS, contrast = _make_simple_inputs()
    LS_bad = np.ones((4, 3))  # 3 cols, X has 4
    with pytest.raises(ValueError, match="columns"):
        estimate_difference(Y, X, LS_bad, contrast)


# --- estimate_difference: contrast index out of bounds ---
def test_estimate_difference_contrast_oob():
    Y, X, LS, contrast = _make_simple_inputs()
    bad_contrast = [[0, 1], [2, 99]]  # index 99 is out of bounds
    with pytest.raises(ValueError, match="index 99"):
        estimate_difference(Y, X, LS, bad_contrast)


# --- estimate_difference: NaN in Y ---
def test_estimate_difference_nan_in_Y():
    Y, X, LS, contrast = _make_simple_inputs()
    Y[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        estimate_difference(Y, X, LS, contrast)


# --- estimate_difference: Inf in model_matrix ---
def test_estimate_difference_inf_in_model_matrix():
    Y, X, LS, contrast = _make_simple_inputs()
    X[0, 0] = np.inf
    with pytest.raises(ValueError, match="Inf"):
        estimate_difference(Y, X, LS, contrast)


# --- RRPP: model_reduced row mismatch ---
def test_rrpp_reduced_row_mismatch():
    Y, X, LS, contrast = _make_simple_inputs()
    X_red = X[:8, :3]  # wrong number of rows
    with pytest.raises(ValueError, match="model_reduced"):
        RRPP(Y, X, X_red, LS, contrast, permutations=2)


# --- RRPP: NaN in model_reduced ---
def test_rrpp_nan_in_model_reduced():
    Y, X, LS, contrast = _make_simple_inputs()
    X_red = X[:, :3].copy()
    X_red[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        RRPP(Y, X, X_red, LS, contrast, permutations=2)
```

- [ ] **Step 2: Run tests to confirm they all fail**

```bash
MOTCO_TEST_PERMS=99 uv run pytest tests/test_validation.py -v
```

Expected: all 7 tests **FAIL** with `AssertionError` or no error raised.

- [ ] **Step 3: Add validation to `estimate_difference` in `sd.py`**

At the very start of the `estimate_difference` body (line ~295, before `n_groups = len(contrast)`), insert:

```python
    # --- Input validation ---
    _Y = np.asarray(Y, dtype=float)
    _X = np.asarray(model_matrix, dtype=float)
    _LS = np.asarray(LS_means, dtype=float)
    if _Y.shape[0] != _X.shape[0]:
        raise ValueError(
            f"Y has {_Y.shape[0]} rows but model_matrix has {_X.shape[0]} rows — "
            "number of rows must match."
        )
    if _LS.shape[1] != _X.shape[1]:
        raise ValueError(
            f"LS_means has {_LS.shape[1]} columns but model_matrix has {_X.shape[1]} columns — "
            "number of columns must match."
        )
    _n_ls = _LS.shape[0]
    for _gi, _group in enumerate(contrast):
        for _idx in _group:
            if not (0 <= _idx < _n_ls):
                raise ValueError(
                    f"contrast[{_gi}] contains index {_idx}, but LS_means only has {_n_ls} rows "
                    f"(valid indices: 0–{_n_ls - 1})."
                )
    if not np.all(np.isfinite(_Y)):
        raise ValueError("Y contains NaN or Inf values.")
    if not np.all(np.isfinite(_X)):
        raise ValueError("model_matrix contains NaN or Inf values.")
    # --- End validation ---
```

- [ ] **Step 4: Add validation to `RRPP` in `sd.py`**

At the very start of the `RRPP` body (line ~365, before `Y = pd.DataFrame(Y)`), insert:

```python
    # --- Input validation ---
    _Y = np.asarray(Y, dtype=float)
    _Xf = np.asarray(model_full, dtype=float)
    _Xr = np.asarray(model_reduced, dtype=float)
    _LS = np.asarray(LS_means, dtype=float)
    if _Y.shape[0] != _Xf.shape[0]:
        raise ValueError(
            f"Y has {_Y.shape[0]} rows but model_full has {_Xf.shape[0]} rows — "
            "number of rows must match."
        )
    if _Y.shape[0] != _Xr.shape[0]:
        raise ValueError(
            f"Y has {_Y.shape[0]} rows but model_reduced has {_Xr.shape[0]} rows — "
            "number of rows must match."
        )
    if _LS.shape[1] != _Xf.shape[1]:
        raise ValueError(
            f"LS_means has {_LS.shape[1]} columns but model_full has {_Xf.shape[1]} columns — "
            "number of columns must match."
        )
    _n_ls = _LS.shape[0]
    for _gi, _group in enumerate(contrast):
        for _idx in _group:
            if not (0 <= _idx < _n_ls):
                raise ValueError(
                    f"contrast[{_gi}] contains index {_idx}, but LS_means only has {_n_ls} rows "
                    f"(valid indices: 0–{_n_ls - 1})."
                )
    if not np.all(np.isfinite(_Y)):
        raise ValueError("Y contains NaN or Inf values.")
    if not np.all(np.isfinite(_Xf)):
        raise ValueError("model_full contains NaN or Inf values.")
    if not np.all(np.isfinite(_Xr)):
        raise ValueError("model_reduced contains NaN or Inf values.")
    # --- End validation ---
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
MOTCO_TEST_PERMS=99 uv run pytest tests/test_validation.py -v
```

Expected: all 7 tests **PASS**.

Also confirm existing tests still pass:

```bash
MOTCO_TEST_PERMS=99 uv run pytest tests/test_sd_smoke.py -v
```

- [ ] **Step 6: Commit**

```bash
git add tests/test_validation.py src/motco/stats/sd.py
git commit -m "feat: add input validation to estimate_difference and RRPP"
```

---

## Task 2: Validate `get_model_matrix` and `center_matrix` inputs in `sd.py`

**Files:**
- Modify: `src/motco/stats/sd.py` (lines 84, 41)
- Modify: `tests/test_validation.py`

- [ ] **Step 1: Append failing tests to `tests/test_validation.py`**

```python
from motco.stats.sd import get_model_matrix, center_matrix


# --- get_model_matrix: missing column ---
def test_get_model_matrix_missing_group_col():
    X = pd.DataFrame({"group": ["A", "B", "A", "B"], "level": ["t0", "t1", "t0", "t1"]})
    with pytest.raises(ValueError, match="'missing'"):
        get_model_matrix(X, group_col="missing", level_col="level")


def test_get_model_matrix_missing_level_col():
    X = pd.DataFrame({"group": ["A", "B", "A", "B"], "level": ["t0", "t1", "t0", "t1"]})
    with pytest.raises(ValueError, match="'missing'"):
        get_model_matrix(X, group_col="group", level_col="missing")


# --- get_model_matrix: single unique value ---
def test_get_model_matrix_single_group():
    X = pd.DataFrame({"group": ["A", "A", "A"], "level": ["t0", "t1", "t0"]})
    with pytest.raises(ValueError, match="unique"):
        get_model_matrix(X, group_col="group", level_col="level")


def test_get_model_matrix_single_level():
    X = pd.DataFrame({"group": ["A", "B", "A"], "level": ["t0", "t0", "t0"]})
    with pytest.raises(ValueError, match="unique"):
        get_model_matrix(X, group_col="group", level_col="level")


# --- center_matrix: missing feature column ---
def test_center_matrix_missing_feature_col():
    dat = pd.DataFrame({
        "group": ["A", "A", "B", "B"],
        "level": ["t0", "t1", "t0", "t1"],
        "f1": [1.0, 2.0, 3.0, 4.0],
    })
    with pytest.raises(ValueError, match="'ghost_col'"):
        center_matrix(dat, group_col="group", level_col="level", feature_cols=["ghost_col"])
```

- [ ] **Step 2: Run new tests to confirm they fail**

```bash
MOTCO_TEST_PERMS=99 uv run pytest tests/test_validation.py::test_get_model_matrix_missing_group_col tests/test_validation.py::test_get_model_matrix_missing_level_col tests/test_validation.py::test_get_model_matrix_single_group tests/test_validation.py::test_get_model_matrix_single_level tests/test_validation.py::test_center_matrix_missing_feature_col -v
```

Expected: all 5 **FAIL**.

- [ ] **Step 3: Add validation to `get_model_matrix` in `sd.py`**

At the start of `get_model_matrix` body (line ~119, before `g_levels = sorted(...)`), insert:

```python
    for col, param in [(group_col, "group_col"), (level_col, "level_col")]:
        if col not in X.columns:
            raise ValueError(
                f"{param}='{col}' not found in X. Available columns: {list(X.columns)}."
            )
        n_unique = X[col].nunique()
        if n_unique < 2:
            raise ValueError(
                f"{param}='{col}' has {n_unique} unique value(s); at least 2 are required."
            )
```

- [ ] **Step 4: Add validation to `center_matrix` in `sd.py`**

At the start of `center_matrix` body (line ~67, before `datc = dat.copy()`), insert:

```python
    if feature_cols is not None:
        missing = [c for c in feature_cols if c not in dat.columns]
        if missing:
            raise ValueError(
                f"feature_cols contains column(s) not found in dat: {missing}."
            )
```

- [ ] **Step 5: Run all validation tests**

```bash
MOTCO_TEST_PERMS=99 uv run pytest tests/test_validation.py -v
```

Expected: all 12 tests **PASS**.

- [ ] **Step 6: Commit**

```bash
git add tests/test_validation.py src/motco/stats/sd.py
git commit -m "feat: add input validation to get_model_matrix and center_matrix"
```

---

## Task 3: Validate `plsda_doubleCV` inputs in `pls.py`

**Files:**
- Modify: `src/motco/stats/pls.py` (line 22)
- Modify: `tests/test_validation.py`

- [ ] **Step 1: Append failing tests to `tests/test_validation.py`**

```python
from motco.stats.pls import plsda_doubleCV


def _small_pls_args():
    """Minimal valid args for plsda_doubleCV (fast: 2-fold, 1 repeat, 2 components)."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((20, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(["A"] * 10 + ["B"] * 10)
    return X, y


# --- plsda_doubleCV: row mismatch ---
def test_plsda_row_mismatch():
    X, y = _small_pls_args()
    y_bad = y.iloc[:18]  # 18 labels, X has 20 rows
    with pytest.raises(ValueError, match="20 rows"):
        plsda_doubleCV(X, y_bad, cv1_splits=2, cv2_splits=2, n_repeats=1, max_components=2)


# --- plsda_doubleCV: single class ---
def test_plsda_single_class():
    X, _ = _small_pls_args()
    y_one = pd.Series(["A"] * 20)
    with pytest.raises(ValueError, match="class"):
        plsda_doubleCV(X, y_one, cv1_splits=2, cv2_splits=2, n_repeats=1, max_components=2)


# --- plsda_doubleCV: max_components too large ---
def test_plsda_max_components_exceeds_features():
    X, y = _small_pls_args()  # X has 5 features
    with pytest.raises(ValueError, match="max_components"):
        plsda_doubleCV(X, y, cv1_splits=2, cv2_splits=2, n_repeats=1, max_components=100)


# --- plsda_doubleCV: NaN in X ---
def test_plsda_nan_in_X():
    X, y = _small_pls_args()
    X.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        plsda_doubleCV(X, y, cv1_splits=2, cv2_splits=2, n_repeats=1, max_components=2)
```

- [ ] **Step 2: Run new tests to confirm they fail**

```bash
MOTCO_TEST_PERMS=99 uv run pytest tests/test_validation.py::test_plsda_row_mismatch tests/test_validation.py::test_plsda_single_class tests/test_validation.py::test_plsda_max_components_exceeds_features tests/test_validation.py::test_plsda_nan_in_X -v
```

Expected: all 4 **FAIL**.

- [ ] **Step 3: Add validation to `plsda_doubleCV` in `pls.py`**

At the very start of `plsda_doubleCV` body (line ~65, before `encoder = OneHotEncoder(...)`), insert:

```python
    _X_arr = np.asarray(X, dtype=float)
    _y_arr = np.asarray(y)
    _n_x = _X_arr.shape[0]
    _n_y = _y_arr.shape[0]
    if _n_x != _n_y:
        raise ValueError(
            f"X has {_n_x} rows but y has {_n_y} rows — number of rows must match."
        )
    _n_classes = len(np.unique(_y_arr))
    if _n_classes < 2:
        raise ValueError(
            f"y has {_n_classes} unique class(es); at least 2 are required."
        )
    if max_components > _X_arr.shape[1]:
        raise ValueError(
            f"max_components={max_components} exceeds the number of features in X "
            f"({_X_arr.shape[1]}); reduce max_components."
        )
    if not np.all(np.isfinite(_X_arr)):
        raise ValueError("X contains NaN or Inf values.")
```

- [ ] **Step 4: Run all validation tests**

```bash
MOTCO_TEST_PERMS=99 uv run pytest tests/test_validation.py -v
```

Expected: all 16 tests **PASS**.

- [ ] **Step 5: Commit**

```bash
git add tests/test_validation.py src/motco/stats/pls.py
git commit -m "feat: add input validation to plsda_doubleCV"
```

---

## Task 4: Validate `get_affinity_matrix` and `SNF` inputs in `snf.py`

**Files:**
- Modify: `src/motco/stats/snf.py` (lines 16, 70)
- Modify: `tests/test_validation.py`

- [ ] **Step 1: Append failing tests to `tests/test_validation.py`**

```python
from motco.stats.snf import get_affinity_matrix, SNF


def _square_affinity(n=10):
    rng = np.random.default_rng(0)
    W = rng.uniform(0, 1, (n, n))
    return (W + W.T) / 2  # symmetric


# --- get_affinity_matrix: misaligned datasets ---
def test_affinity_matrix_row_mismatch():
    rng = np.random.default_rng(0)
    d1 = rng.standard_normal((10, 5))
    d2 = rng.standard_normal((8, 5))  # different row count
    with pytest.raises(ValueError, match="8 rows"):
        get_affinity_matrix([d1, d2], K=3)


# --- get_affinity_matrix: K >= n_samples ---
def test_affinity_matrix_K_too_large():
    rng = np.random.default_rng(0)
    d1 = rng.standard_normal((5, 3))
    d2 = rng.standard_normal((5, 3))
    with pytest.raises(ValueError, match="K=10"):
        get_affinity_matrix([d1, d2], K=10)


# --- get_affinity_matrix: eps <= 0 ---
def test_affinity_matrix_invalid_eps():
    rng = np.random.default_rng(0)
    d1 = rng.standard_normal((10, 3))
    d2 = rng.standard_normal((10, 3))
    with pytest.raises(ValueError, match="eps"):
        get_affinity_matrix([d1, d2], K=3, eps=-0.1)


# --- get_affinity_matrix: NaN in data ---
def test_affinity_matrix_nan():
    rng = np.random.default_rng(0)
    d1 = rng.standard_normal((10, 3))
    d2 = rng.standard_normal((10, 3))
    d2[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        get_affinity_matrix([d1, d2], K=3)


# --- SNF: non-square matrix ---
def test_snf_non_square():
    W1 = _square_affinity(5)
    W2 = np.ones((5, 6))  # not square
    with pytest.raises(ValueError, match="not square"):
        SNF([W1, W2])


# --- SNF: shape mismatch ---
def test_snf_shape_mismatch():
    W1 = _square_affinity(5)
    W2 = _square_affinity(6)
    with pytest.raises(ValueError, match="shape"):
        SNF([W1, W2])


# --- SNF: k >= n_samples ---
def test_snf_k_too_large():
    W1 = _square_affinity(5)
    W2 = _square_affinity(5)
    with pytest.raises(ValueError, match="k=10"):
        SNF([W1, W2], k=10)
```

- [ ] **Step 2: Run new tests to confirm they fail**

```bash
MOTCO_TEST_PERMS=99 uv run pytest tests/test_validation.py::test_affinity_matrix_row_mismatch tests/test_validation.py::test_affinity_matrix_K_too_large tests/test_validation.py::test_affinity_matrix_invalid_eps tests/test_validation.py::test_affinity_matrix_nan tests/test_validation.py::test_snf_non_square tests/test_validation.py::test_snf_shape_mismatch tests/test_validation.py::test_snf_k_too_large -v
```

Expected: all 7 **FAIL**.

- [ ] **Step 3: Add validation to `get_affinity_matrix` in `snf.py`**

At the start of `get_affinity_matrix` body (line ~90, before `Ws: list[np.ndarray] = []`), insert:

```python
    if not dats:
        raise ValueError("dats must contain at least one dataset.")
    _n = np.asarray(dats[0]).shape[0]
    for _i, _dat in enumerate(dats):
        _arr = np.asarray(_dat)
        if _arr.shape[0] != _n:
            raise ValueError(
                f"dats[{_i}] has {_arr.shape[0]} rows but dats[0] has {_n} rows — "
                "all datasets must have the same number of rows."
            )
        if not np.all(np.isfinite(_arr)):
            raise ValueError(f"dats[{_i}] contains NaN or Inf values.")
    if K >= _n:
        raise ValueError(
            f"K={K} must be less than the number of samples ({_n})."
        )
    if eps <= 0:
        raise ValueError(f"eps={eps} must be positive.")
```

- [ ] **Step 4: Add validation to `SNF` in `snf.py`**

Replace the existing `if nw < 2` check (line ~40) with the full validation block:

```python
    nw = len(Ws)
    if nw < 2:
        raise ValueError("SNF requires at least two affinity matrices.")
    _n = Ws[0].shape[0]
    for _i, _W in enumerate(Ws):
        if _W.shape[0] != _W.shape[1]:
            raise ValueError(
                f"Ws[{_i}] is not square: shape {_W.shape}."
            )
        if _W.shape[0] != _n:
            raise ValueError(
                f"Ws[{_i}] has shape {_W.shape} but Ws[0] has shape {Ws[0].shape} — "
                "all matrices must have the same shape."
            )
    if k >= _n:
        raise ValueError(
            f"k={k} must be less than the number of samples ({_n})."
        )
```

- [ ] **Step 5: Run all validation tests**

```bash
MOTCO_TEST_PERMS=99 uv run pytest tests/test_validation.py -v
```

Expected: all 23 tests **PASS**.

Also run the full fast suite:

```bash
MOTCO_TEST_PERMS=99 uv run pytest tests/test_sd_smoke.py tests/test_validation.py -v
```

- [ ] **Step 6: Commit**

```bash
git add tests/test_validation.py src/motco/stats/snf.py
git commit -m "feat: add input validation to get_affinity_matrix and SNF"
```

---

## Task 5: Create `tests/test_pls.py` smoke tests

**Files:**
- Create: `tests/test_pls.py`

- [ ] **Step 1: Create `tests/test_pls.py`**

```python
# tests/test_pls.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from motco.stats.pls import plsda_doubleCV


def _synthetic_data(n: int = 30, p: int = 10, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, p)), columns=[f"f{i}" for i in range(p)])
    y = pd.Series(["A"] * (n // 2) + ["B"] * (n - n // 2))
    return X, y


def test_plsda_returns_required_keys():
    X, y = _synthetic_data()
    result = plsda_doubleCV(X, y, cv1_splits=3, cv2_splits=3, n_repeats=2, max_components=3)
    assert set(result.keys()) >= {"table", "models"}


def test_plsda_table_has_correct_row_count():
    n_repeats = 3
    X, y = _synthetic_data()
    result = plsda_doubleCV(X, y, cv1_splits=3, cv2_splits=3, n_repeats=n_repeats, max_components=3)
    assert result["table"].shape[0] == n_repeats


def test_plsda_auroc_values_in_unit_interval():
    X, y = _synthetic_data()
    result = plsda_doubleCV(X, y, cv1_splits=3, cv2_splits=3, n_repeats=2, max_components=3)
    aurocs = result["table"].iloc[:, 2].values  # AUROC is third column
    assert np.all(aurocs >= 0.0) and np.all(aurocs <= 1.0)


def test_plsda_model_count_matches_repeats():
    n_repeats = 2
    X, y = _synthetic_data()
    result = plsda_doubleCV(X, y, cv1_splits=3, cv2_splits=3, n_repeats=n_repeats, max_components=3)
    assert len(result["models"]) == n_repeats


def test_plsda_lv_values_are_positive_integers():
    X, y = _synthetic_data()
    result = plsda_doubleCV(X, y, cv1_splits=3, cv2_splits=3, n_repeats=2, max_components=3)
    lvs = result["table"].iloc[:, 1].values  # LV is second column
    assert np.all(lvs >= 1)
    assert np.all(lvs == lvs.astype(int))
```

- [ ] **Step 2: Run the new tests**

```bash
MOTCO_TEST_PERMS=99 uv run pytest tests/test_pls.py -v
```

Expected: all 5 tests **PASS** (PLS already works; these tests confirm the interface).

- [ ] **Step 3: Commit**

```bash
git add tests/test_pls.py
git commit -m "test: add PLS smoke tests"
```

---

## Task 6: Create `tests/test_snf.py` smoke tests

**Files:**
- Create: `tests/test_snf.py`

- [ ] **Step 1: Create `tests/test_snf.py`**

```python
# tests/test_snf.py
from __future__ import annotations

import numpy as np
import pytest

from motco.stats.snf import get_affinity_matrix, SNF, get_spectral


def _datasets(n: int = 20, p: int = 5, n_datasets: int = 2, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.standard_normal((n, p)) for _ in range(n_datasets)]


def test_affinity_matrix_returns_correct_count():
    dats = _datasets()
    Ws = get_affinity_matrix(dats, K=5)
    assert len(Ws) == len(dats)


def test_affinity_matrix_shapes_are_square():
    dats = _datasets(n=20)
    Ws = get_affinity_matrix(dats, K=5)
    for W in Ws:
        assert W.shape == (20, 20)


def test_affinity_matrix_non_negative():
    dats = _datasets()
    Ws = get_affinity_matrix(dats, K=5)
    for W in Ws:
        assert np.all(W >= 0)


def test_affinity_matrix_symmetric():
    dats = _datasets()
    Ws = get_affinity_matrix(dats, K=5)
    for W in Ws:
        assert np.allclose(W, W.T)


def test_snf_output_shape():
    n = 20
    dats = _datasets(n=n)
    Ws = get_affinity_matrix(dats, K=5)
    fused = SNF(Ws, k=5, t=5)
    assert fused.shape == (n, n)


def test_snf_output_symmetric():
    dats = _datasets()
    Ws = get_affinity_matrix(dats, K=5)
    fused = SNF(Ws, k=5, t=5)
    assert np.allclose(fused, fused.T)


def test_get_spectral_default_shape():
    dats = _datasets(n=20)
    Ws = get_affinity_matrix(dats, K=5)
    fused = SNF(Ws, k=5, t=5)
    emb = get_spectral(fused)
    assert emb.shape == (20, 10)  # default 10 components
```

- [ ] **Step 2: Run the new tests**

```bash
MOTCO_TEST_PERMS=99 uv run pytest tests/test_snf.py -v
```

Expected: all 7 tests **PASS**.

- [ ] **Step 3: Commit**

```bash
git add tests/test_snf.py
git commit -m "test: add SNF smoke tests"
```

---

## Task 7: Create `tests/test_cli.py` CLI tests

**Files:**
- Create: `tests/test_cli.py`

- [ ] **Step 1: Create `tests/test_cli.py`**

```python
# tests/test_cli.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from motco.cli import main
from motco.stats.sd import build_ls_means, get_model_matrix


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def plsr_csv(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    n, p = 30, 5
    df = pd.DataFrame(rng.standard_normal((n, p)), columns=[f"f{i}" for i in range(p)])
    df["label"] = ["A"] * 15 + ["B"] * 15
    path = tmp_path / "plsr_data.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def snf_csvs(tmp_path: Path) -> list[Path]:
    rng = np.random.default_rng(0)
    paths = []
    for i in range(2):
        df = pd.DataFrame(rng.standard_normal((15, 4)))
        p = tmp_path / f"omics{i}.csv"
        df.to_csv(p, index=False)
        paths.append(p)
    return paths


@pytest.fixture()
def de_files(tmp_path: Path) -> dict[str, Path]:
    rng = np.random.default_rng(0)
    n = 20
    Y = pd.DataFrame(rng.standard_normal((n, 3)))
    factors = pd.DataFrame({
        "group": ["A"] * 10 + ["B"] * 10,
        "level": ["t0", "t1"] * 10,
    })
    M_full = get_model_matrix(factors, group_col="group", level_col="level", full=True)
    M_red = get_model_matrix(factors, group_col="group", level_col="level", full=False)
    LS = build_ls_means(["A", "B"], ["t0", "t1"], full=True)
    contrast = [[0, 1], [2, 3]]

    paths = {}
    paths["Y"] = tmp_path / "Y.csv"
    paths["model"] = tmp_path / "model.csv"
    paths["model_full"] = tmp_path / "model_full.csv"
    paths["model_red"] = tmp_path / "model_red.csv"
    paths["ls"] = tmp_path / "ls.csv"
    paths["contrast"] = tmp_path / "contrast.json"

    Y.to_csv(paths["Y"], index=False)
    pd.DataFrame(M_full).to_csv(paths["model"], index=False)
    pd.DataFrame(M_full).to_csv(paths["model_full"], index=False)
    pd.DataFrame(M_red).to_csv(paths["model_red"], index=False)
    pd.DataFrame(LS).to_csv(paths["ls"], index=False)
    paths["contrast"].write_text(json.dumps(contrast))
    return paths


# ── plsr subcommand ───────────────────────────────────────────────────────────

def test_plsr_saves_csv(tmp_path: Path, plsr_csv: Path) -> None:
    out = tmp_path / "table.csv"
    main([
        "plsr", "--data", str(plsr_csv), "--label-col", "label",
        "--cv1-splits", "3", "--cv2-splits", "3", "--n-repeats", "2",
        "--max-components", "3", "--out-table", str(out),
    ])
    assert out.exists()
    df = pd.read_csv(out)
    assert df.shape[0] == 2  # n_repeats rows


def test_plsr_prints_to_stdout(capsys: pytest.CaptureFixture, plsr_csv: Path) -> None:
    main([
        "plsr", "--data", str(plsr_csv), "--label-col", "label",
        "--cv1-splits", "3", "--cv2-splits", "3", "--n-repeats", "2",
        "--max-components", "3",
    ])
    out = capsys.readouterr().out
    assert "AUROC" in out


def test_plsr_bad_label_col_exits(plsr_csv: Path) -> None:
    with pytest.raises(SystemExit):
        main([
            "plsr", "--data", str(plsr_csv), "--label-col", "nonexistent",
            "--cv1-splits", "3", "--cv2-splits", "3", "--n-repeats", "2",
            "--max-components", "3",
        ])


# ── snf subcommand ────────────────────────────────────────────────────────────

def test_snf_saves_fused_matrix(tmp_path: Path, snf_csvs: list[Path]) -> None:
    out = tmp_path / "fused.csv"
    main([
        "snf",
        "--input", str(snf_csvs[0]), "--input", str(snf_csvs[1]),
        "--K", "5", "--k", "5", "--t", "3", "--out-fused", str(out),
    ])
    assert out.exists()
    df = pd.read_csv(out)
    assert df.shape == (15, 15)


def test_snf_saves_embedding(tmp_path: Path, snf_csvs: list[Path]) -> None:
    out_fused = tmp_path / "fused.csv"
    out_emb = tmp_path / "emb.csv"
    main([
        "snf",
        "--input", str(snf_csvs[0]), "--input", str(snf_csvs[1]),
        "--K", "5", "--k", "5", "--t", "3",
        "--out-fused", str(out_fused), "--out-embedding", str(out_emb),
    ])
    assert out_emb.exists()
    emb = pd.read_csv(out_emb)
    assert emb.shape == (15, 10)


def test_snf_requires_two_inputs() -> None:
    with pytest.raises(SystemExit):
        main(["snf", "--input", "only_one.csv"])


# ── de subcommand — estimate path ─────────────────────────────────────────────

def test_de_estimate_saves_json(tmp_path: Path, de_files: dict[str, Path]) -> None:
    out = tmp_path / "result.json"
    main([
        "de",
        "--Y", str(de_files["Y"]),
        "--model-matrix", str(de_files["model"]),
        "--ls-means", str(de_files["ls"]),
        "--contrast", str(de_files["contrast"]),
        "--out-json", str(out),
    ])
    assert out.exists()
    result = json.loads(out.read_text())
    assert {"deltas", "angles", "shapes"} <= result.keys()
    # 2 groups → 2×2 matrices
    assert len(result["deltas"]) == 2
    assert len(result["deltas"][0]) == 2


# ── de subcommand — RRPP path ─────────────────────────────────────────────────

def test_de_rrpp_saves_json(tmp_path: Path, de_files: dict[str, Path]) -> None:
    out = tmp_path / "rrpp.json"
    main([
        "de",
        "--Y", str(de_files["Y"]),
        "--model-full", str(de_files["model_full"]),
        "--model-reduced", str(de_files["model_red"]),
        "--ls-means", str(de_files["ls"]),
        "--contrast", str(de_files["contrast"]),
        "--rrpp-permutations", "2",
        "--out-json", str(out),
    ])
    assert out.exists()
    result = json.loads(out.read_text())
    assert {"deltas", "angles", "shapes"} <= result.keys()
    assert len(result["deltas"]) == 2  # 2 permutations
```

- [ ] **Step 2: Run the new tests**

```bash
MOTCO_TEST_PERMS=99 uv run pytest tests/test_cli.py -v
```

Expected: all 10 tests **PASS**.

- [ ] **Step 3: Commit**

```bash
git add tests/test_cli.py
git commit -m "test: add CLI subcommand tests"
```

---

## Task 8: Register `slow` marker and tag regression tests

**Files:**
- Modify: `pyproject.toml`
- Modify: `tests/test_sd_expected_example1.py`
- Modify: `tests/test_sd_expected_example2.py`

- [ ] **Step 1: Register the marker in `pyproject.toml`**

In the `[tool.pytest.ini_options]` section, add `markers`:

```toml
[tool.pytest.ini_options]
testpaths = [
  "tests",
]
python_files = [
  "test_*.py",
  "*_test.py",
]
addopts = "-ra -v -s"
markers = [
  "slow: marks tests as slow (deselect with -m 'not slow'; use MOTCO_TEST_PERMS=10000 for full run)",
]
```

- [ ] **Step 2: Add `@pytest.mark.slow` to `test_sd_expected_example1.py`**

Add the import and decorator:

```python
import pytest  # add this import at the top

@pytest.mark.slow
def test_example1_expected_results_match(data_dir):
    ...
```

- [ ] **Step 3: Add `@pytest.mark.slow` to `test_sd_expected_example2.py`**

Same change as Step 2 but in `test_sd_expected_example2.py`.

- [ ] **Step 4: Verify fast tests exclude slow tests**

```bash
MOTCO_TEST_PERMS=99 uv run pytest tests/ -m "not slow" -v --collect-only
```

Expected: the two `test_example*_expected_results_match` tests do **not** appear in the collected list.

- [ ] **Step 5: Verify slow tests are selectable**

```bash
MOTCO_TEST_PERMS=99 uv run pytest tests/ -m "slow" --collect-only
```

Expected: exactly 2 tests collected.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml tests/test_sd_expected_example1.py tests/test_sd_expected_example2.py
git commit -m "test: register slow marker and tag regression tests"
```

---

## Task 9: Add `--version` and `--out-observed` to CLI

**Files:**
- Modify: `src/motco/cli.py`

- [ ] **Step 1: Write failing tests for the new CLI flags**

Append to `tests/test_cli.py`:

```python
# ── --version flag ────────────────────────────────────────────────────────────

def test_version_flag_prints_version(capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    # argparse prints version to stdout
    assert "motco" in capsys.readouterr().out


# ── --out-observed flag ───────────────────────────────────────────────────────

def test_de_out_observed_saves_csv(tmp_path: Path, de_files: dict[str, Path]) -> None:
    out_json = tmp_path / "result.json"
    out_obs = tmp_path / "observed.csv"
    main([
        "de",
        "--Y", str(de_files["Y"]),
        "--model-matrix", str(de_files["model"]),
        "--ls-means", str(de_files["ls"]),
        "--contrast", str(de_files["contrast"]),
        "--out-json", str(out_json),
        "--out-observed", str(out_obs),
    ])
    assert out_obs.exists()
    obs = pd.read_csv(out_obs)
    # LS has 4 rows (2 groups × 2 levels), Y has 3 features
    assert obs.shape == (4, 3)
```

- [ ] **Step 2: Run new tests to confirm they fail**

```bash
MOTCO_TEST_PERMS=99 uv run pytest tests/test_cli.py::test_version_flag_prints_version tests/test_cli.py::test_de_out_observed_saves_csv -v
```

Expected: both **FAIL**.

- [ ] **Step 3: Add `--version` to `build_parser` in `cli.py`**

In `build_parser`, after `p = argparse.ArgumentParser(...)`, add:

```python
from motco import __version__  # add at top of file with other imports

# In build_parser, after creating p:
p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
```

- [ ] **Step 4: Add `--out-observed` and `estimate_betas` to `cli.py`**

At the top of `cli.py`, update the `sd` import:

```python
from motco.stats.sd import RRPP, estimate_difference, estimate_betas
```

In `build_parser`, add to the `de` subparser (after `--out-json`):

```python
p_de.add_argument("--out-observed", type=str, help="Save predicted LS-mean vectors as CSV")
```

In `cmd_de`, for the non-RRPP branch, add after `_save_json(out, args.out_json)` (or the `else` print):

```python
    if args.out_observed:
        betas = estimate_betas(X, Y)
        observed = np.asarray(LS, dtype=float) @ np.asarray(betas, dtype=float)
        _save_csv(observed, args.out_observed)
```

For the RRPP branch, add after RRPP output is serialized:

```python
    if args.out_observed:
        betas = estimate_betas(Xf, Y)
        observed = np.asarray(LS, dtype=float) @ np.asarray(betas, dtype=float)
        _save_csv(observed, args.out_observed)
```

Note: in `cmd_de`, `Y` is already a numpy array (`_read_csv(args.Y).values`). `estimate_betas` accepts numpy arrays and returns a numpy array when given one.

- [ ] **Step 5: Wrap stats calls in try/except in all three subcommands**

The validation in the stats modules raises `ValueError` with descriptive messages. Without wrapping, these appear as Python tracebacks for external users. Add try/except in each command function to convert them to clean `SystemExit` messages.

In `cmd_plsr`, wrap the `plsda_doubleCV` call:

```python
    try:
        res = plsda_doubleCV(
            X=X, y=y,
            cv1_splits=args.cv1_splits, cv2_splits=args.cv2_splits,
            n_repeats=args.n_repeats, max_components=args.max_components,
            random_state=args.random_state,
        )
    except ValueError as e:
        raise SystemExit(f"Error: {e}") from None
```

In `cmd_snf`, wrap the `get_affinity_matrix` and `SNF` calls:

```python
    try:
        Ws = get_affinity_matrix(datasets, K=args.K, eps=args.eps)
        fused = SNF(Ws, k=args.k, t=args.t)
    except ValueError as e:
        raise SystemExit(f"Error: {e}") from None
```

In `cmd_de`, wrap both the `estimate_difference` and `RRPP` calls:

```python
    try:
        if args.rrpp_permutations and args.rrpp_permutations > 0:
            ...  # RRPP call
        else:
            ...  # estimate_difference call
    except ValueError as e:
        raise SystemExit(f"Error: {e}") from None
```

- [ ] **Step 6: Run all CLI tests**

```bash
MOTCO_TEST_PERMS=99 uv run pytest tests/test_cli.py -v
```

Expected: all 12 tests **PASS**.

- [ ] **Step 7: Commit**

```bash
git add src/motco/cli.py tests/test_cli.py
git commit -m "feat: add --version, --out-observed, and clean error messages to CLI"
```

---

## Task 10: Update README

**Files:**
- Modify: `README.md`
- Modify: `pyproject.toml`

- [ ] **Step 1: Fix placeholder homepage URL in `pyproject.toml`**

In `[project.urls]`, replace the placeholder:

```toml
[project.urls]
Homepage = "https://github.com/tomszar/MOTCO"
```

- [ ] **Step 2: Add "Interpreting your results" section to README**

After the `## Python API (selected)` section, add:

```markdown
## Interpreting Results

`estimate_difference` and `RRPP` return three symmetric matrices:

| Output | Meaning |
|--------|---------|
| `deltas` | Absolute difference in trajectory magnitude (total path length) between group pairs. Larger = one group changed more than the other. |
| `angles` | Angle in degrees between trajectory orientations. 0° = same direction; 90° = orthogonal; 180° = exactly opposite. |
| `shapes` | Procrustes distance between trajectory shapes after removing size and orientation differences. 0 = identical shape. |

**P-values via RRPP:** Use a right-tailed test with the add-one correction:

```python
def pvalue(samples, observed, i, j):
    vals = np.array([s[i, j] for s in samples])
    return (np.sum(vals >= observed) + 1) / (len(vals) + 1)
```

Significance threshold is typically α = 0.05.
```

- [ ] **Step 3: Add "Quick Example" section pointing to `examples/`**

After the install section and before "Command Line Interface", add:

```markdown
## Quick Example

See `examples/motco_example.ipynb` for an end-to-end walkthrough using the bundled dataset.

The equivalent CLI commands:

```bash
# 1. Build latent space with PLS-DA
motco plsr --data tests/data/evo_649_sm_example1.csv --label-col taxa \
  --cv1-splits 7 --cv2-splits 8 --n-repeats 5 --max-components 2 \
  --out-table results/plsr_table.csv

# 2. Estimate group differences
motco de \
  --Y results/latent_space.csv \
  --model-matrix results/model_matrix.csv \
  --ls-means results/ls_means.csv \
  --contrast contrast.json \
  --out-json results/de_result.json \
  --out-observed results/ls_mean_vectors.csv
```
```

- [ ] **Step 4: Document `get_observed_vectors` in the Python API section of README**

In the existing `## Python API (selected)` section, add after the current import list:

```markdown
### Inspecting LS-mean coordinates

Before running `estimate_difference`, use `get_observed_vectors` to see the predicted
mean position of each group × level cell in Y space:

```python
from motco.stats.sd import get_observed_vectors

# X_factors: DataFrame with group_col and level_col
# Y: outcome matrix aligned to X_factors by row
obs = get_observed_vectors(X_factors, Y, group_col='group', level_col='level', full=True)
# Returns a DataFrame with MultiIndex (group, level) and columns matching Y
print(obs)
```
```

- [ ] **Step 5: Commit**

```bash
git add README.md pyproject.toml
git commit -m "docs: add results interpretation, quick example, fix homepage URL"
```

---

## Task 11: Create example notebook `examples/motco_example.ipynb`

**Files:**
- Create: `examples/motco_example.ipynb`

- [ ] **Step 1: Create the `examples/` directory and generate the notebook**

Run this Python script once to produce the `.ipynb` file:

```bash
mkdir -p examples
python - <<'SCRIPT'
import json, pathlib

cells = [
    {"cell_type": "markdown", "source": "# MOTCO End-to-End Example\n\nThis notebook walks through the complete MOTCO workflow using the bundled `evo_649_sm_example1.csv` dataset.\n\n**Dataset:** 180 samples, 2 features (V1, V2), group column `taxa`, level column `Inv`.", "metadata": {}},

    {"cell_type": "code", "source": "import numpy as np\nimport pandas as pd\nfrom pathlib import Path\nfrom sklearn.decomposition import PCA\n\nfrom motco.stats.sd import (\n    get_model_matrix, build_ls_means, estimate_difference,\n    RRPP, get_observed_vectors, center_matrix,\n)\n\n# Load the bundled example dataset\ndata_path = Path('../tests/data/evo_649_sm_example1.csv')\ndf = pd.read_csv(data_path)\nprint(df.head())\nprint(f'Shape: {df.shape}, Groups: {df[\"taxa\"].unique()}, Levels: {df[\"Inv\"].unique()}')", "metadata": {}, "outputs": [], "execution_count": None},

    {"cell_type": "markdown", "source": "## Step 1 — Define column roles\n\n- `group_col`: the between-subject factor (which taxon / group the sample belongs to)\n- `level_col`: the within-group factor (which timepoint / state the sample is at)\n- Feature columns: all remaining numeric columns — these are the outcome `Y`", "metadata": {}},

    {"cell_type": "code", "source": "group_col = 'taxa'\nlevel_col = 'Inv'\nfeat_cols = [c for c in df.select_dtypes(include=[float]).columns if c not in {group_col, level_col}]\n\n# Reduce to 2 PCA dimensions as the outcome space\npca = PCA(n_components=2, random_state=0)\nY = pd.DataFrame(pca.fit_transform(df[feat_cols]), columns=['PC1', 'PC2'])\nprint(f'Y shape: {Y.shape}')\nprint(f'Variance explained: {pca.explained_variance_ratio_}')", "metadata": {}, "outputs": [], "execution_count": None},

    {"cell_type": "markdown", "source": "## Step 2 — Build the model matrix and LS means\n\n`get_model_matrix` creates a design matrix with:\n- An intercept column\n- Dummy variables for groups (drop-first coding)\n- Dummy variables for levels (drop-first coding)\n- Group × level interaction terms (when `full=True`)\n\n`build_ls_means` creates one row per group × level cell, in the same column coding. These rows, multiplied by the regression coefficients, give the least-squares mean vector for each cell.", "metadata": {}},

    {"cell_type": "code", "source": "factors = df[[group_col, level_col]].copy()\ng_levels = sorted(df[group_col].astype(str).unique())\nl_levels = sorted(df[level_col].astype(str).unique())\nprint(f'Groups: {g_levels}')\nprint(f'Levels: {l_levels}')\n\nM_full = get_model_matrix(factors, group_col=group_col, level_col=level_col, full=True)\nM_red  = get_model_matrix(factors, group_col=group_col, level_col=level_col, full=False)\nLS = build_ls_means(g_levels, l_levels, full=True)\n\nprint(f'Full model matrix shape: {M_full.shape}')\nprint(f'Reduced model matrix shape: {M_red.shape}')\nprint(f'LS means shape: {LS.shape}  (one row per group×level cell)')", "metadata": {}, "outputs": [], "execution_count": None},

    {"cell_type": "markdown", "source": "## Step 3 — Inspect LS-mean vectors\n\n`get_observed_vectors` fits the full model and returns the predicted mean position (in Y space) for each group × level combination.", "metadata": {}},

    {"cell_type": "code", "source": "obs = get_observed_vectors(factors, Y, group_col=group_col, level_col=level_col, full=True)\nprint(obs)", "metadata": {}, "outputs": [], "execution_count": None},

    {"cell_type": "markdown", "source": "## Step 4 — Define the contrast\n\nThe `contrast` list tells `estimate_difference` which LS-mean rows belong to each group's trajectory.\n\nRow order in `LS` is group-major, level-minor:\n- Row 0: group[0], level[0]\n- Row 1: group[0], level[1]\n- Row 2: group[1], level[0]\n- Row 3: group[1], level[1]\n- …", "metadata": {}},

    {"cell_type": "code", "source": "L = len(l_levels)\ncontrast = [[gi * L + li for li in range(L)] for gi in range(len(g_levels))]\nprint('Contrast:', contrast)\nprint('Meaning:')\nfor gi, group in enumerate(g_levels):\n    print(f'  Group {group}: LS-mean rows {contrast[gi]} → levels {l_levels}')", "metadata": {}, "outputs": [], "execution_count": None},

    {"cell_type": "markdown", "source": "## Step 5 — Estimate trajectory differences\n\n`estimate_difference` returns three symmetric matrices comparing every pair of groups:\n- `deltas`: difference in trajectory magnitude (total path length)\n- `angles`: angle in degrees between trajectory directions (0°=same, 90°=orthogonal, 180°=opposite)\n- `shapes`: Procrustes distance between trajectory shapes", "metadata": {}},

    {"cell_type": "code", "source": "deltas, angles, shapes = estimate_difference(Y, M_full, LS, contrast)\ngroup_labels = g_levels\nprint('Angles (degrees):')\nprint(pd.DataFrame(angles, index=group_labels, columns=group_labels).round(2))\nprint('\\nDeltas (magnitude difference):')\nprint(pd.DataFrame(deltas, index=group_labels, columns=group_labels).round(4))", "metadata": {}, "outputs": [], "execution_count": None},

    {"cell_type": "markdown", "source": "## Step 6 — RRPP permutation test\n\nRRPP permutes the residuals of the reduced model to build a null distribution for each statistic. The p-value is right-tailed with the add-one correction.\n\n**Note:** Set `MOTCO_TEST_PERMS=999` or higher for a real analysis. We use 99 here for speed.", "metadata": {}},

    {"cell_type": "code", "source": "import os\nPERMS = int(os.getenv('MOTCO_NOTEBOOK_PERMS', '99'))\n\ndist_delta, dist_angle, _ = RRPP(Y, M_full, M_red, LS, contrast, permutations=PERMS)\n\ndef pvalue(samples, observed, i, j):\n    vals = np.array([s[i, j] for s in samples])\n    return (np.sum(vals >= observed) + 1) / (len(vals) + 1)\n\nprint(f'Results ({PERMS} permutations):')\nfor i, g1 in enumerate(g_levels):\n    for j, g2 in enumerate(g_levels):\n        if j <= i:\n            continue\n        ang = angles[i, j]\n        dlt = deltas[i, j]\n        p_ang = pvalue(dist_angle, ang, i, j)\n        p_dlt = pvalue(dist_delta, dlt, i, j)\n        print(f'  {g1} vs {g2}: angle={ang:.2f}° (p={p_ang:.3f}), delta={dlt:.4f} (p={p_dlt:.3f})')", "metadata": {}, "outputs": [], "execution_count": None},

    {"cell_type": "markdown", "source": "## Equivalent CLI commands\n\nThe analysis above maps to:\n\n```bash\n# Estimate differences\nmotco de \\\n  --Y Y.csv \\\n  --model-matrix model_full.csv \\\n  --ls-means ls_means.csv \\\n  --contrast contrast.json \\\n  --out-json result.json \\\n  --out-observed ls_mean_vectors.csv\n\n# With RRPP\nmotco de \\\n  --Y Y.csv \\\n  --model-full model_full.csv \\\n  --model-reduced model_reduced.csv \\\n  --ls-means ls_means.csv \\\n  --contrast contrast.json \\\n  --rrpp-permutations 999 \\\n  --out-json rrpp_result.json\n```", "metadata": {}},
]

def make_cell(c):
    if c['cell_type'] == 'markdown':
        return {'cell_type': 'markdown', 'metadata': {}, 'source': c['source']}
    return {
        'cell_type': 'code', 'execution_count': None, 'metadata': {},
        'outputs': [], 'source': c['source'],
    }

nb = {
    'nbformat': 4, 'nbformat_minor': 5,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.11.0'},
    },
    'cells': [make_cell(c) for c in cells],
}

pathlib.Path('examples/motco_example.ipynb').write_text(json.dumps(nb, indent=1))
print('Notebook written to examples/motco_example.ipynb')
SCRIPT
```

- [ ] **Step 2: Verify the notebook is valid JSON and can be opened**

```bash
python -c "import json; nb = json.load(open('examples/motco_example.ipynb')); print(f'{len(nb[\"cells\"])} cells OK')"
```

Expected: `13 cells OK`

- [ ] **Step 3: Commit**

```bash
git add examples/motco_example.ipynb
git commit -m "docs: add end-to-end example notebook"
```

---

## Task 12: Expose `calculate_vips` and add `--out-vips` to CLI

**Files:**
- Modify: `src/motco/stats/pls.py`
- Modify: `src/motco/cli.py`
- Modify: `tests/test_pls.py`

- [ ] **Step 1: Add failing test for `calculate_vips` to `tests/test_pls.py`**

```python
from motco.stats.pls import plsda_doubleCV, calculate_vips


def test_calculate_vips_shape():
    X, y = _synthetic_data()
    result = plsda_doubleCV(X, y, cv1_splits=3, cv2_splits=3, n_repeats=2, max_components=3)
    model = result["models"][0]
    vips = calculate_vips(model)
    assert vips.shape == (X.shape[1],)  # one VIP per feature


def test_calculate_vips_non_negative():
    X, y = _synthetic_data()
    result = plsda_doubleCV(X, y, cv1_splits=3, cv2_splits=3, n_repeats=2, max_components=3)
    model = result["models"][0]
    vips = calculate_vips(model)
    assert np.all(vips >= 0)
```

- [ ] **Step 2: Run failing tests to confirm**

```bash
MOTCO_TEST_PERMS=99 uv run pytest tests/test_pls.py::test_calculate_vips_shape tests/test_pls.py::test_calculate_vips_non_negative -v
```

Expected: both **FAIL** with `ImportError` (name `calculate_vips` not found).

- [ ] **Step 3: Rename `_calculate_vips` → `calculate_vips` in `pls.py`**

On line 185, change the function definition:

```python
def calculate_vips(  # was: def _calculate_vips(
```

Update the docstring first line accordingly. No other changes needed — the function body is unchanged.

- [ ] **Step 4: Run pls tests to confirm they pass**

```bash
MOTCO_TEST_PERMS=99 uv run pytest tests/test_pls.py -v
```

Expected: all 7 tests **PASS**.

- [ ] **Step 5: Add failing CLI test for `--out-vips`**

Append to `tests/test_cli.py`:

```python
def test_plsr_out_vips_saves_csv(tmp_path: Path, plsr_csv: Path) -> None:
    out_table = tmp_path / "table.csv"
    out_vips = tmp_path / "vips.csv"
    main([
        "plsr", "--data", str(plsr_csv), "--label-col", "label",
        "--cv1-splits", "3", "--cv2-splits", "3", "--n-repeats", "2",
        "--max-components", "3",
        "--out-table", str(out_table),
        "--out-vips", str(out_vips),
    ])
    assert out_vips.exists()
    vips_df = pd.read_csv(out_vips)
    # 5 features (from plsr_csv fixture), 2 repeats
    assert vips_df.shape == (5, 2)
    assert list(vips_df.columns) == ["rep_1", "rep_2"]
```

- [ ] **Step 6: Run CLI test to confirm it fails**

```bash
MOTCO_TEST_PERMS=99 uv run pytest tests/test_cli.py::test_plsr_out_vips_saves_csv -v
```

Expected: **FAIL** (argument not yet added).

- [ ] **Step 7: Add `--out-vips` to `cmd_plsr` and `build_parser` in `cli.py`**

Update the `plsr` import at top of `cli.py`:

```python
from motco.stats.pls import plsda_doubleCV, calculate_vips
```

In `build_parser`, add to the `plsr` subparser (after `--out-table`):

```python
p_plsr.add_argument("--out-vips", type=str, help="Path to save VIP scores per feature (CSV)")
```

In `cmd_plsr`, after saving/printing the table, add:

```python
    if args.out_vips:
        vips_data = {}
        for rep_idx, model in enumerate(res["models"], start=1):
            vips_data[f"rep_{rep_idx}"] = calculate_vips(model)
        vips_df = pd.DataFrame(vips_data)
        _save_csv(vips_df, args.out_vips)
```

- [ ] **Step 8: Run all CLI tests**

```bash
MOTCO_TEST_PERMS=99 uv run pytest tests/test_cli.py -v
```

Expected: all 13 tests **PASS**.

- [ ] **Step 9: Commit**

```bash
git add src/motco/stats/pls.py src/motco/cli.py tests/test_pls.py tests/test_cli.py
git commit -m "feat: expose calculate_vips and add --out-vips to motco plsr"
```

---

## Task 13: Add `n_components` to `get_spectral` and `--spectral-components` to CLI

**Files:**
- Modify: `src/motco/stats/snf.py`
- Modify: `src/motco/cli.py`
- Modify: `tests/test_snf.py`

- [ ] **Step 1: Add failing test to `tests/test_snf.py`**

```python
def test_get_spectral_custom_components():
    dats = _datasets(n=20)
    Ws = get_affinity_matrix(dats, K=5)
    fused = SNF(Ws, k=5, t=5)
    emb = get_spectral(fused, n_components=5)
    assert emb.shape == (20, 5)
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
MOTCO_TEST_PERMS=99 uv run pytest tests/test_snf.py::test_get_spectral_custom_components -v
```

Expected: **FAIL** with `TypeError` (unexpected keyword argument).

- [ ] **Step 3: Update `get_spectral` signature in `snf.py`**

Change the function signature from:

```python
def get_spectral(aff: np.ndarray) -> np.ndarray:
```

To:

```python
def get_spectral(aff: np.ndarray, n_components: int = 10) -> np.ndarray:
```

And update the body from:

```python
    embedding = spectral_embedding(aff, n_components=10, random_state=1548)
```

To:

```python
    embedding = spectral_embedding(aff, n_components=n_components, random_state=1548)
```

- [ ] **Step 4: Run all SNF tests**

```bash
MOTCO_TEST_PERMS=99 uv run pytest tests/test_snf.py -v
```

Expected: all 8 tests **PASS** (default behavior preserved).

- [ ] **Step 5: Add failing CLI test**

Append to `tests/test_cli.py`:

```python
def test_snf_custom_spectral_components(tmp_path: Path, snf_csvs: list[Path]) -> None:
    out_fused = tmp_path / "fused.csv"
    out_emb = tmp_path / "emb.csv"
    main([
        "snf",
        "--input", str(snf_csvs[0]), "--input", str(snf_csvs[1]),
        "--K", "5", "--k", "5", "--t", "3",
        "--out-fused", str(out_fused),
        "--out-embedding", str(out_emb),
        "--spectral-components", "5",
    ])
    assert out_emb.exists()
    emb = pd.read_csv(out_emb)
    assert emb.shape == (15, 5)
```

- [ ] **Step 6: Run CLI test to confirm it fails**

```bash
MOTCO_TEST_PERMS=99 uv run pytest tests/test_cli.py::test_snf_custom_spectral_components -v
```

Expected: **FAIL** (argument not yet added).

- [ ] **Step 7: Add `--spectral-components` to `build_parser` and `cmd_snf` in `cli.py`**

In `build_parser`, add to the `snf` subparser (after `--out-embedding`):

```python
p_snf.add_argument("--spectral-components", type=int, default=10,
                   help="Number of spectral embedding components (default: 10)")
```

In `cmd_snf`, update the `get_spectral` call:

```python
        emb = get_spectral(fused, n_components=args.spectral_components)
```

- [ ] **Step 8: Run all CLI tests**

```bash
MOTCO_TEST_PERMS=99 uv run pytest tests/test_cli.py -v
```

Expected: all 14 tests **PASS**.

- [ ] **Step 9: Commit**

```bash
git add src/motco/stats/snf.py src/motco/cli.py tests/test_snf.py tests/test_cli.py
git commit -m "feat: add n_components to get_spectral and --spectral-components to motco snf"
```

---

## Task 14: Extend `stats/__init__.py` with remaining public functions

**Files:**
- Modify: `src/motco/stats/__init__.py`
- Modify: `tests/test_pls.py` (add one import test)

- [ ] **Step 1: Add failing import test to `tests/test_pls.py`**

```python
def test_stats_top_level_imports():
    from motco.stats import (  # noqa: F401
        estimate_difference, RRPP, get_model_matrix, build_ls_means,
        get_observed_vectors, pair_difference, center_matrix, estimate_betas,
        plsda_doubleCV, calculate_vips,
        get_affinity_matrix, SNF, get_spectral,
    )
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
MOTCO_TEST_PERMS=99 uv run pytest tests/test_pls.py::test_stats_top_level_imports -v
```

Expected: **FAIL** with `ImportError`.

- [ ] **Step 3: Update `src/motco/stats/__init__.py`**

Replace the entire file content with:

```python
"""Statistical utilities for MOTCO.

Modules
-------
pls
    Partial Least Squares (PLS-DA) utilities.
snf
    Similarity Network Fusion and spectral embedding.
sd
    Trajectory group differences (delta, angle, shape) and RRPP.
"""

from .pls import plsda_doubleCV, calculate_vips  # noqa: F401
from .sd import (  # noqa: F401
    RRPP,
    build_ls_means,
    center_matrix,
    estimate_betas,
    estimate_difference,
    get_model_matrix,
    get_observed_vectors,
    pair_difference,
)
from .snf import SNF, get_affinity_matrix, get_spectral  # noqa: F401

__all__ = [
    # pls
    "plsda_doubleCV",
    "calculate_vips",
    # sd
    "RRPP",
    "build_ls_means",
    "center_matrix",
    "estimate_betas",
    "estimate_difference",
    "get_model_matrix",
    "get_observed_vectors",
    "pair_difference",
    # snf
    "SNF",
    "get_affinity_matrix",
    "get_spectral",
]
```

- [ ] **Step 4: Run all tests**

```bash
MOTCO_TEST_PERMS=99 uv run pytest tests/ -m "not slow" -v
```

Expected: all tests **PASS** (no regressions).

- [ ] **Step 5: Run lint and type check**

```bash
uv run ruff check src/ tests/
uv run mypy src/motco/
```

Fix any issues before committing.

- [ ] **Step 6: Commit**

```bash
git add src/motco/stats/__init__.py tests/test_pls.py
git commit -m "feat: export all public functions from motco.stats"
```

---

## Final Verification

- [ ] Run the full fast test suite and confirm it passes:

```bash
MOTCO_TEST_PERMS=99 uv run pytest tests/ -m "not slow" -v
```

- [ ] Run lint and type checks:

```bash
uv run ruff check src/ tests/
uv run mypy src/motco/
```

- [ ] Spot-check the notebook runs (requires jupyter):

```bash
cd examples && jupyter nbconvert --to notebook --execute motco_example.ipynb --output motco_example_executed.ipynb 2>&1 | tail -5
```

- [ ] Push:

```bash
git push
```
