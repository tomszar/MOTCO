"""Tests for src/motco/viz.py."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.decomposition import PCA

matplotlib.use("Agg")


def _make_viz_inputs(
    n_per_cell: int = 8,
    n_features: int = 10,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (Y, metadata) for a 2-group × 2-stage design."""
    rng = np.random.default_rng(seed)
    groups = ["A", "B"]
    stages = ["s0", "s1"]
    rows = []
    for g in groups:
        for s in stages:
            offset = np.array([1.0, 0.0] + [0.0] * (n_features - 2)) * (
                2.0 if g == "B" else 0.0
            ) + np.array([0.0, 1.0] + [0.0] * (n_features - 2)) * (
                1.0 if s == "s1" else 0.0
            )
            for _ in range(n_per_cell):
                features = {f"f{i}": float(v) for i, v in enumerate(rng.standard_normal(n_features) + offset)}
                rows.append({"group": g, "stage": s, **features})
    df = pd.DataFrame(rows)
    metadata = df[["group", "stage"]]
    Y = df.drop(columns=["group", "stage"])
    return Y, metadata


# ── 5.4 Top-level import ──────────────────────────────────────────────────────

def test_top_level_import():
    from motco import plot_trajectories, plot_trajectory_from_data  # noqa: F401


# ── 5.1 Smoke test: plot_trajectories ─────────────────────────────────────────

def test_plot_trajectories_smoke():
    from motco import plot_trajectories

    Y, metadata = _make_viz_inputs()
    pca = PCA(n_components=2).fit(Y.values)

    groups = metadata["group"].unique().tolist()
    stages = metadata["stage"].unique().tolist()
    idx = pd.MultiIndex.from_product([sorted(groups), sorted(stages)], names=["group", "stage"])
    observed_vectors = pd.DataFrame(
        pca.transform(Y.values)[:len(idx)],
        index=idx,
        columns=[f"pc{i}" for i in range(2)],
    )
    # Use a properly shaped observed_vectors via the real helper
    from motco.stats.trajectory import get_observed_vectors
    observed_vectors = get_observed_vectors(metadata, Y, group_col="group", level_col="stage")

    fig, ax = plot_trajectories(
        observed_vectors=observed_vectors,
        projector=pca,
        group_col="group",
        level_col="stage",
    )

    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)


# ── 5.2 Smoke test: plot_trajectory_from_data ─────────────────────────────────

def test_plot_trajectory_from_data_smoke():
    from motco import plot_trajectory_from_data

    Y, metadata = _make_viz_inputs()
    fig, ax, pca = plot_trajectory_from_data(
        Y=Y,
        metadata=metadata,
        group_col="group",
        level_col="stage",
    )

    assert isinstance(fig, Figure)
    assert isinstance(pca, PCA)
    assert hasattr(pca, "explained_variance_ratio_")

    import matplotlib.pyplot as plt
    plt.close(fig)


# ── 5.5 PLS projector: plot_trajectory_from_plsr ──────────────────────────────

def test_plot_trajectory_from_plsr_smoke():
    from sklearn.cross_decomposition import PLSRegression

    from motco import plot_trajectory_from_plsr

    Y, metadata = _make_viz_inputs()
    fig, ax, pls = plot_trajectory_from_plsr(
        Y=Y,
        metadata=metadata,
        group_col="group",
        level_col="stage",
    )

    assert isinstance(fig, Figure)
    assert isinstance(pls, PLSRegression)
    # PLS projector relabels axes away from the PCA default.
    assert ax.get_xlabel().startswith("PLS1")

    import matplotlib.pyplot as plt
    plt.close(fig)


# ── 5.3 show_samples=True path ────────────────────────────────────────────────

def test_plot_trajectories_show_samples():
    from motco import plot_trajectory_from_data

    Y, metadata = _make_viz_inputs()
    fig, ax, _ = plot_trajectory_from_data(
        Y=Y,
        metadata=metadata,
        group_col="group",
        level_col="stage",
        show_samples=True,
    )

    # Scatter artists should be present (PathCollection objects)
    from matplotlib.collections import PathCollection
    scatter_artists = [c for c in ax.collections if isinstance(c, PathCollection)]
    assert len(scatter_artists) > 0, "Expected scatter artists when show_samples=True"

    import matplotlib.pyplot as plt
    plt.close(fig)
