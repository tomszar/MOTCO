"""Visualization utilities for MOTCO trajectory geometry."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA

from motco.stats.pls import fit_plsda_model
from motco.stats.trajectory import get_observed_vectors


def plot_trajectories(
    observed_vectors: pd.DataFrame,
    projector: Any,
    ax: Axes | None = None,
    show_samples: bool = False,
    samples: pd.DataFrame | None = None,
    sample_metadata: pd.DataFrame | None = None,
    group_col: str = "group",
    level_col: str = "level",
    palette: dict[str, Any] | None = None,
    component_label: str = "PC",
) -> tuple[Figure, Axes]:
    """Plot trajectory geometry from pre-computed LS-mean vectors.

    Parameters
    ----------
    observed_vectors:
        MultiIndex DataFrame (group, level) of LS-mean vectors.
        Typically the output of ``get_observed_vectors()``.
    projector:
        Fitted projector with a ``.transform()`` method (e.g. sklearn PCA).
    ax:
        Existing axes to draw on. Created if not provided.
    show_samples:
        If True, overlay individual sample points behind the trajectories.
        Requires ``samples`` and ``sample_metadata``.
    samples:
        Raw outcome matrix (n_samples × n_features) for scatter overlay.
    sample_metadata:
        DataFrame with at least a ``group_col`` column aligned to ``samples``.
    group_col:
        Column name for the group factor in ``sample_metadata`` and the
        observed_vectors index.
    level_col:
        Column name for the stage/level factor in the observed_vectors index.
    palette:
        Mapping from group label to matplotlib color. Falls back to the
        default matplotlib color cycle when not provided.
    component_label:
        Prefix for the projected-axis labels (e.g. ``"PC"`` for PCA, ``"PLS"``
        for a PLS projector). Defaults to ``"PC"``.

    Returns
    -------
    tuple[Figure, Axes]
    """
    groups = observed_vectors.index.get_level_values(group_col).unique().tolist()
    stages = observed_vectors.index.get_level_values(level_col).unique().tolist()

    if palette is None:
        prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        palette = {g: prop_cycle[i % len(prop_cycle)] for i, g in enumerate(groups)}

    if ax is None:
        fig, ax = plt.subplots()
    else:
        raw_fig = ax.get_figure()
        if not isinstance(raw_fig, Figure):
            raise TypeError("ax must belong to a Figure, not a SubFigure")
        fig = raw_fig

    # --- Optional sample scatter ---
    if show_samples and samples is not None and sample_metadata is not None:
        projected_samples = projector.transform(np.asarray(samples, dtype=float))
        for group in groups:
            mask = sample_metadata[group_col].astype(str) == str(group)
            ax.scatter(
                projected_samples[mask, 0],
                projected_samples[mask, 1],
                color=palette[group],
                alpha=0.3,
                s=20,
                linewidths=0,
                zorder=1,
            )

    # --- Trajectory paths ---
    for group in groups:
        color = palette[group]
        pts = observed_vectors.loc[group]
        projected = projector.transform(np.asarray(pts, dtype=float))

        # Draw connected line
        ax.plot(projected[:, 0], projected[:, 1], color=color, linewidth=1.5, zorder=2)

        # Draw LS-mean points
        ax.scatter(projected[:, 0], projected[:, 1], color=color, s=60, zorder=3, label=str(group))

        # Direction arrows at each segment midpoint
        for i in range(len(projected) - 1):
            start = projected[i]
            end = projected[i + 1]
            mid = (start + end) / 2.0
            direction = end - start
            ax.annotate(
                "",
                xy=mid + direction * 0.01,
                xytext=mid - direction * 0.01,
                arrowprops=dict(
                    arrowstyle="->",
                    color=color,
                    lw=1.5,
                ),
                zorder=4,
            )

        # Label the first stage point
        first_stage = str(stages[0])
        ax.annotate(
            first_stage,
            xy=(projected[0, 0], projected[0, 1]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            color=color,
            zorder=5,
        )

    # --- Axis labels with explained variance ---
    xlabel = f"{component_label}1"
    ylabel = f"{component_label}2"
    if hasattr(projector, "explained_variance_ratio_"):
        evr = projector.explained_variance_ratio_
        if len(evr) >= 2:
            xlabel = f"{component_label}1 ({evr[0] * 100:.1f}%)"
            ylabel = f"{component_label}2 ({evr[1] * 100:.1f}%)"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(title=group_col)

    return fig, ax


def plot_trajectory_from_data(
    Y: pd.DataFrame,
    metadata: pd.DataFrame,
    group_col: str,
    level_col: str,
    full: bool = True,
    n_components: int = 2,
    ax: Axes | None = None,
    show_samples: bool = False,
    palette: dict[str, Any] | None = None,
) -> tuple[Figure, Axes, PCA]:
    """Fit PCA on outcome matrix and plot trajectory geometry.

    Convenience wrapper around ``plot_trajectories`` that handles PCA fitting
    internally. The fitted PCA is returned so callers can reuse the same
    coordinate system for additional figures.

    Parameters
    ----------
    Y:
        Outcome matrix (n_samples × n_features).
    metadata:
        Sample metadata DataFrame with at least ``group_col`` and ``level_col``
        columns, row-aligned with ``Y``.
    group_col:
        Column name for the group factor.
    level_col:
        Column name for the stage/level factor.
    full:
        Whether to include group × level interactions in the model matrix
        passed to ``get_observed_vectors``.
    n_components:
        Number of PCA components to fit. Must be at least 2.
    ax:
        Existing axes to draw on. Created if not provided.
    show_samples:
        If True, overlay individual sample scatter behind the trajectories.
    palette:
        Mapping from group label to matplotlib color.

    Returns
    -------
    tuple[Figure, Axes, PCA]
        The figure, axes, and fitted PCA object.
    """
    observed_vectors = get_observed_vectors(
        metadata[[group_col, level_col]],
        Y,
        group_col=group_col,
        level_col=level_col,
        full=full,
    )

    pca = PCA(n_components=n_components)
    pca.fit(np.asarray(Y, dtype=float))

    fig, ax = plot_trajectories(
        observed_vectors=observed_vectors,
        projector=pca,
        ax=ax,
        show_samples=show_samples,
        samples=Y if show_samples else None,
        sample_metadata=metadata if show_samples else None,
        group_col=group_col,
        level_col=level_col,
        palette=palette,
    )

    return fig, ax, pca


def plot_trajectory_from_plsr(
    Y: pd.DataFrame,
    metadata: pd.DataFrame,
    group_col: str,
    level_col: str,
    full: bool = True,
    n_components: int = 2,
    ax: Axes | None = None,
    show_samples: bool = True,
    palette: dict[str, Any] | None = None,
) -> tuple[Figure, Axes, PLSRegression]:
    """Fit a 2-component PLS-DA on ``Y`` and plot trajectory geometry.

    PLS analog of :func:`plot_trajectory_from_data`. The projector is a
    supervised PLS-DA fit with the stage/level factor as the response (one-hot
    encoded), so the latent axes are oriented to separate stages rather than to
    maximize total variance. No cross-validation is performed — the model is fit
    once with a fixed ``n_components``. The fitted PLS estimator is returned so
    callers can reuse the same coordinate system for additional figures.

    Parameters
    ----------
    Y:
        Outcome matrix (n_samples × n_features).
    metadata:
        Sample metadata DataFrame with at least ``group_col`` and ``level_col``
        columns, row-aligned with ``Y``.
    group_col:
        Column name for the group factor.
    level_col:
        Column name for the stage/level factor. Also used as the PLS response.
    full:
        Whether to include group × level interactions in the model matrix
        passed to ``get_observed_vectors``.
    n_components:
        Number of PLS latent variables to fit. Must be at least 2.
    ax:
        Existing axes to draw on. Created if not provided.
    show_samples:
        If True, overlay individual sample scatter behind the trajectories.
    palette:
        Mapping from group label to matplotlib color.

    Returns
    -------
    tuple[Figure, Axes, PLSRegression]
        The figure, axes, and fitted PLS estimator.
    """
    observed_vectors = get_observed_vectors(
        metadata[[group_col, level_col]],
        Y,
        group_col=group_col,
        level_col=level_col,
        full=full,
    )

    pls = fit_plsda_model(Y, metadata[level_col], n_components=n_components)

    fig, ax = plot_trajectories(
        observed_vectors=observed_vectors,
        projector=pls,
        ax=ax,
        show_samples=show_samples,
        samples=Y if show_samples else None,
        sample_metadata=metadata if show_samples else None,
        group_col=group_col,
        level_col=level_col,
        palette=palette,
        component_label="PLS",
    )

    return fig, ax, pls
