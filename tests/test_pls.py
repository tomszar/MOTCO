# tests/test_pls.py
from __future__ import annotations

import numpy as np
import pandas as pd

from motco.stats.pls import calculate_vips, fit_plsda_transform, plsda_doubleCV


def _synthetic_data(n: int = 30, p: int = 10, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, p)), columns=[f"f{i}" for i in range(p)])
    y = pd.Series(["A"] * (n // 2) + ["B"] * (n - n // 2))
    return X, y


def test_plsda_returns_required_keys():
    X, y = _synthetic_data()
    result = plsda_doubleCV(X, y, cv1_splits=3, cv2_splits=3, n_repeats=2, max_components=3)
    assert set(result.keys()) >= {"table", "models"}


def test_plsda_table_has_correct_shape():
    n_repeats = 3
    X, y = _synthetic_data()
    result = plsda_doubleCV(X, y, cv1_splits=3, cv2_splits=3, n_repeats=n_repeats, max_components=3)
    assert result["table"].shape == (n_repeats, 4)
    assert list(result["table"].columns) == ["rep", "LV", "AUROC", "AUROC_std"]


def test_plsda_auroc_values_in_unit_interval():
    X, y = _synthetic_data()
    result = plsda_doubleCV(X, y, cv1_splits=3, cv2_splits=3, n_repeats=2, max_components=3)
    aurocs = result["table"].iloc[:, 2].values  # AUROC is third column
    assert np.all(aurocs >= 0.0) and np.all(aurocs <= 1.0)


def test_plsda_auroc_std_non_negative():
    X, y = _synthetic_data()
    result = plsda_doubleCV(X, y, cv1_splits=3, cv2_splits=3, n_repeats=2, max_components=3)
    stds = result["table"]["AUROC_std"].values
    finite = stds[~np.isnan(stds)]
    assert np.all(finite >= 0.0)


def test_plsda_doubleCV_mean_aggregation_is_honest():
    """Synthetic data with moderate SNR: mean AUROC should be < 0.99 with K outer folds.

    Group A ~ N(0, I_20); Group B ~ N(mu, I_20) with mu = 0.5 in 20 features.
    Bayes-optimal AUROC ≈ 0.94; small-n CV variance produces visible per-fold spread.
    Under the old max-of-K aggregation, at least one fold per repeat would saturate
    at AUROC = 1.0, pinning the reported value. Mean-of-K reporting produces an
    honest summary.
    """
    rng = np.random.default_rng(42)
    n_per = 30
    X_A = rng.standard_normal((n_per, 20))
    X_B = rng.standard_normal((n_per, 20)) + 0.5
    X = pd.DataFrame(np.vstack([X_A, X_B]), columns=[f"f{i}" for i in range(20)])
    y = pd.Series(["A"] * n_per + ["B"] * n_per)
    result = plsda_doubleCV(X, y, cv1_splits=3, cv2_splits=4, n_repeats=3, max_components=5)
    aurocs = result["table"]["AUROC"].values
    auroc_stds = result["table"]["AUROC_std"].values
    assert np.mean(aurocs) < 0.99, (
        f"Mean AUROC {np.mean(aurocs):.4f} is suspiciously close to 1.0; "
        "max-of-K aggregation may have leaked through."
    )
    assert np.mean(auroc_stds) > 0.0, (
        "AUROC_std is zero across repeats; expected fold-to-fold variance "
        "with moderate-SNR data."
    )


def test_plsda_doubleCV_models_refit_on_full_data():
    """Models stored per repeat are full-data refits, not CV-fold models."""
    n_repeats = 2
    X, y = _synthetic_data(n=30, p=10)
    result = plsda_doubleCV(
        X, y, cv1_splits=3, cv2_splits=3, n_repeats=n_repeats, max_components=3
    )
    for i in range(n_repeats):
        model = result["models"][i]
        assert model.x_scores_.shape[0] == X.shape[0], (
            f"models[{i}].x_scores_.shape[0]={model.x_scores_.shape[0]}, "
            f"expected full-data refit on {X.shape[0]} samples"
        )
        # Per-repeat n_components should match the modal LV recorded in the table.
        assert model.n_components == int(result["table"].iloc[i, 1])


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


def test_fit_plsda_transform_shape():
    X, y = _synthetic_data(n=30, p=10)
    scores = fit_plsda_transform(X, y, n_components=3)
    assert scores.shape == (30, 3)


def test_fit_plsda_transform_single_component():
    X, y = _synthetic_data(n=30, p=10)
    scores = fit_plsda_transform(X, y, n_components=1)
    assert scores.shape == (30, 1)


def test_fit_plsda_transform_returns_ndarray():
    X, y = _synthetic_data(n=30, p=10)
    scores = fit_plsda_transform(X, y, n_components=2)
    assert isinstance(scores, np.ndarray)


def test_fit_plsda_transform_numeric_labels():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((30, 8)), columns=[f"f{i}" for i in range(8)])
    y = pd.Series([0, 1, 2] * 10)
    scores = fit_plsda_transform(X, y, n_components=2)
    assert scores.shape == (30, 2)


def test_stats_top_level_imports():
    from motco.stats import (  # noqa: F401
        RRPP,
        SNF,
        build_ls_means,
        calculate_vips,
        center_matrix,
        estimate_betas,
        estimate_difference,
        fit_plsda_transform,
        get_affinity_matrix,
        get_model_matrix,
        get_observed_vectors,
        get_spectral,
        pair_difference,
        plsda_doubleCV,
    )
