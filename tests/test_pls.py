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
