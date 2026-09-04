from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from motco.stats.design import build_ls_means, center_matrix, get_model_matrix


def _feature_columns(df: pd.DataFrame, group_col: str, level_col: str) -> list[str]:
    return [
        c for c in df.select_dtypes(include=[np.number]).columns if c not in {group_col, level_col}
    ]


# ── Smoke tests ───────────────────────────────────────────────────────────────

def test_center_matrix_group_means_zero(example_df: pd.DataFrame, group_col: str, level_col: str):
    feat_cols = _feature_columns(example_df, group_col, level_col)
    centered = center_matrix(example_df, group_col=group_col, level_col=level_col, feature_cols=feat_cols)

    grp_means = centered.groupby(group_col)[feat_cols].mean()
    assert np.allclose(grp_means.values, 0.0, atol=1e-10)


def test_model_matrix_and_ls_means_shapes(example_df: pd.DataFrame, group_col: str, level_col: str):
    X = example_df[[group_col, level_col]].copy()
    g_levels = sorted(pd.unique(X[group_col].astype(str)).tolist())
    l_levels = sorted(pd.unique(X[level_col].astype(str)).tolist())

    M = get_model_matrix(X, group_col=group_col, level_col=level_col, full=True)
    Gm1 = max(len(g_levels) - 1, 0)
    Lm1 = max(len(l_levels) - 1, 0)
    expected_cols = 1 + Gm1 + Lm1 + (Gm1 * Lm1)
    assert M.shape[0] == len(X)
    assert M.shape[1] == expected_cols

    LS = build_ls_means(g_levels, l_levels, full=True)
    assert LS.shape[0] == len(g_levels) * len(l_levels)
    assert LS.shape[1] == expected_cols


# ── Validation tests ──────────────────────────────────────────────────────────

def test_get_model_matrix_missing_group_col():
    X = pd.DataFrame({"group": ["A", "B", "A", "B"], "level": ["t0", "t1", "t0", "t1"]})
    with pytest.raises(ValueError, match="'missing'"):
        get_model_matrix(X, group_col="missing", level_col="level")


def test_get_model_matrix_missing_level_col():
    X = pd.DataFrame({"group": ["A", "B", "A", "B"], "level": ["t0", "t1", "t0", "t1"]})
    with pytest.raises(ValueError, match="'missing'"):
        get_model_matrix(X, group_col="group", level_col="missing")


def test_get_model_matrix_single_group():
    X = pd.DataFrame({"group": ["A", "A", "A"], "level": ["t0", "t1", "t0"]})
    with pytest.raises(ValueError, match="unique"):
        get_model_matrix(X, group_col="group", level_col="level")


def test_get_model_matrix_single_level():
    X = pd.DataFrame({"group": ["A", "B", "A"], "level": ["t0", "t0", "t0"]})
    with pytest.raises(ValueError, match="unique"):
        get_model_matrix(X, group_col="group", level_col="level")


def test_center_matrix_missing_feature_col():
    dat = pd.DataFrame({
        "group": ["A", "A", "B", "B"],
        "level": ["t0", "t1", "t0", "t1"],
        "f1": [1.0, 2.0, 3.0, 4.0],
    })
    with pytest.raises(ValueError, match="'ghost_col'"):
        center_matrix(dat, group_col="group", level_col="level", feature_cols=["ghost_col"])


# ── Level ordering (audit D8: "10" must not sort before "2") ─────────────────

def test_sort_levels_numeric_order_for_integer_labels():
    from motco.stats.design import _sort_levels

    labels = [str(i) for i in range(12)]
    assert _sort_levels(list(reversed(labels))) == labels


def test_sort_levels_lexicographic_for_non_numeric_labels():
    from motco.stats.design import _sort_levels

    assert _sort_levels(["B", "A"]) == ["A", "B"]
    # Mixed alphanumeric labels stay lexicographic — no general natural sort.
    assert _sort_levels(["stage10", "stage2"]) == ["stage10", "stage2"]


def test_sort_levels_lexicographic_on_numeric_collision():
    from motco.stats.design import _sort_levels

    # "01" and "1" collide numerically; a numeric key would be ill-defined.
    assert _sort_levels(["1", "01"]) == ["01", "1"]


def test_model_matrix_orders_ten_plus_integer_levels_numerically():
    n_stages = 12
    frame = pd.DataFrame(
        {
            "group": ["A", "B"] * n_stages,
            "stage": [str(s) for s in range(n_stages) for _ in (0, 1)],
        }
    )
    model = get_model_matrix(frame, group_col="group", level_col="stage", full=True)
    # Intercept + (G-1) + (L-1) + (G-1)(L-1) columns.
    assert model.shape == (2 * n_stages, 1 + 1 + (n_stages - 1) + (n_stages - 1))
    # Level dummy columns are 2..2+(L-1): drop-first over the ordered levels,
    # so the row for stage "2" must activate the dummy at ordinal position 2
    # (numeric order), not position 4 (lexicographic "0","1","10","11","2").
    row_stage2 = frame.index[frame["stage"] == "2"][0]
    level_dummies = model[row_stage2, 2 : 2 + n_stages - 1]
    assert level_dummies[1] == 1.0  # dummy for level ordinal 2 (first was dropped)
    assert level_dummies.sum() == 1.0


def test_simulation_design_orders_ten_plus_stages_numerically():
    from motco.simulations.evaluation import build_simulation_trajectory_design

    n_stages = 12
    metadata = pd.DataFrame(
        {
            "group": ["A", "B"] * n_stages,
            "stage": [str(s) for s in range(n_stages) for _ in (0, 1)],
        }
    )
    design = build_simulation_trajectory_design(metadata)
    assert design.stage_levels == [str(s) for s in range(n_stages)]
    assert design.group_levels == ["A", "B"]
    # Contrast indices are group-major, level-minor over the numeric order.
    assert design.contrast == [
        list(range(n_stages)),
        list(range(n_stages, 2 * n_stages)),
    ]
    assert design.ls_means.shape[0] == 2 * n_stages


def test_observed_vectors_index_orders_integer_levels_numerically():
    from motco.stats.trajectory import get_observed_vectors

    n_stages = 11
    rng = np.random.default_rng(1203)
    frame = pd.DataFrame(
        {
            "group": ["A", "B"] * n_stages,
            "stage": [str(s) for s in range(n_stages) for _ in (0, 1)],
        }
    )
    Y = pd.DataFrame(rng.normal(size=(2 * n_stages, 3)), columns=["f1", "f2", "f3"])
    means = get_observed_vectors(frame, Y, "group", "stage", full=True)
    stage_order = means.index.get_level_values("stage")[:n_stages].tolist()
    assert stage_order == [str(s) for s in range(n_stages)]
