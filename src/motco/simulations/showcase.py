"""Side-by-side visualization of the ``trajectory_mode`` scenarios.

This module is an illustrative demo (not part of the power study). It generates,
for each ``trajectory_mode`` — ``none`` (null), ``translation``, ``magnitude``,
``orientation``, ``shape`` — a dataset from the numpy generator, reusing the
*same* seed so that group A's baseline trajectory and the group assignment are
identical across panels and only group B's feature-surgery transform differs.
Each scenario is integrated (concatenation across omics), projected through its
own 2-component PLS-DA (stage as the response, no cross-validation), and
rendered as a directed trajectory in a multi-panel figure. No R is required.

Typical entry point: :func:`run_trajectory_showcase`.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from motco.simulations.evaluation import (
    SimulationEvaluationParams,
    integrate_semisynthetic_dataset,
)
from motco.simulations.reference import IntersimReference, load_reference
from motco.simulations.semisynthetic import (
    SemiSyntheticTrajectoryDataset,
    SemiSyntheticTrajectoryParams,
    generate_semisynthetic_trajectory,
)
from motco.viz import plot_trajectory_from_plsr

# Order reads as: null → location → size → orientation → shape.
TRAJECTORY_SHOWCASE_MODES: tuple[str, ...] = (
    "none",
    "translation",
    "magnitude",
    "orientation",
    "shape",
)

# Panel titles annotate each mode with the geometric axis it exercises.
_MODE_TITLES: dict[str, str] = {
    "none": "none — null (no group effect)",
    "translation": "translation — constant offset",
    "magnitude": "magnitude — size (delta)",
    "orientation": "orientation — angle",
    "shape": "shape — localized bend",
}


class TrajectoryShowcaseError(ValueError):
    """Raised when showcase inputs or parameters are invalid."""


def generate_showcase_datasets(
    *,
    modes: tuple[str, ...] = TRAJECTORY_SHOWCASE_MODES,
    effect_size: float = 1.0,
    seed: int = 0,
    n_samples: int = 1000,
    n_stages: int = 3,
    group_ratio: float = 0.5,
    p_dmp: float = 0.2,
    reference: IntersimReference | None = None,
) -> dict[str, SemiSyntheticTrajectoryDataset]:
    """Generate one dataset per ``trajectory_mode`` from a shared baseline.

    The same ``seed`` is used for every mode so that group A's baseline
    trajectory and the within-stage group assignment are identical across
    scenarios; the only thing that changes between datasets is group B's
    feature-surgery transform. ``none`` always gets a zero effect size (the
    null), regardless of ``effect_size``.

    Returns a dict mapping mode name → dataset, preserving the order of ``modes``.
    """
    if not modes:
        raise TrajectoryShowcaseError("modes must contain at least one trajectory mode.")

    ref = reference if reference is not None else load_reference()
    datasets: dict[str, SemiSyntheticTrajectoryDataset] = {}
    for mode in modes:
        params = SemiSyntheticTrajectoryParams(
            seed=seed,
            trajectory_mode=mode,  # type: ignore[arg-type]
            n_samples=n_samples,
            n_stages=n_stages,
            group_effect_size=0.0 if mode == "none" else effect_size,
            group_ratio=group_ratio,
            p_dmp=p_dmp,
        )
        datasets[mode] = generate_semisynthetic_trajectory(params, reference=ref)
    return datasets


def build_showcase_figure(
    datasets: Mapping[str, SemiSyntheticTrajectoryDataset],
    *,
    group_col: str = "group",
    stage_col: str = "stage",
    integration_params: Mapping[str, Any] | None = None,
    n_cols: int = 3,
    show_samples: bool = True,
    figsize_per_panel: tuple[float, float] = (4.5, 4.0),
    suptitle: str | None = "Trajectory modes (per-scenario PLS-DA on stage)",
) -> Figure:
    """Render one PLS-DA trajectory panel per scenario in a single figure.

    Each dataset is concatenation-integrated into an outcome matrix ``Y`` and
    projected through its own 2-component PLS-DA (via
    :func:`motco.viz.plot_trajectory_from_plsr`). Unused grid cells are hidden.
    """
    if not datasets:
        raise TrajectoryShowcaseError("datasets must contain at least one scenario.")
    if n_cols < 1:
        raise TrajectoryShowcaseError("n_cols must be at least 1.")

    eval_params = SimulationEvaluationParams(
        integration_method="concat",
        integration_params=dict(integration_params or {}),
        group_col=group_col,
        stage_col=stage_col,
    )

    n_panels = len(datasets)
    n_cols = min(n_cols, n_panels)
    n_rows = math.ceil(n_panels / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        squeeze=False,
    )
    flat_axes = axes.ravel()

    for i, (mode, dataset) in enumerate(datasets.items()):
        ax = flat_axes[i]
        latent = integrate_semisynthetic_dataset(dataset, eval_params)
        Y = latent.matrix.reset_index(drop=True)
        metadata = dataset.metadata.reset_index(drop=True)
        plot_trajectory_from_plsr(
            Y=Y,
            metadata=metadata,
            group_col=group_col,
            level_col=stage_col,
            ax=ax,
            show_samples=show_samples,
        )
        ax.set_title(_MODE_TITLES.get(mode, mode))

    for ax in flat_axes[n_panels:]:
        ax.set_visible(False)

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def run_trajectory_showcase(
    *,
    seed: int = 0,
    n_sample: int = 1000,
    n_stages: int = 3,
    effect_size: float = 1.0,
    modes: tuple[str, ...] = TRAJECTORY_SHOWCASE_MODES,
    group_ratio: float = 0.5,
    p_dmp: float = 0.2,
    integration_params: Mapping[str, Any] | None = None,
    show_samples: bool = True,
    reference: IntersimReference | None = None,
) -> tuple[Figure, dict[str, SemiSyntheticTrajectoryDataset]]:
    """Generate the trajectory-mode showcase end to end (no R required).

    Generates one dataset per mode in ``modes`` from a shared baseline (cached
    reference data) and renders the comparison figure. Returns
    ``(figure, datasets)`` so callers can save the figure and/or inspect the
    per-scenario datasets and their injected-truth metadata.
    """
    datasets = generate_showcase_datasets(
        modes=modes,
        effect_size=effect_size,
        seed=seed,
        n_samples=n_sample,
        n_stages=n_stages,
        group_ratio=group_ratio,
        p_dmp=p_dmp,
        reference=reference,
    )
    fig = build_showcase_figure(
        datasets,
        integration_params=integration_params,
        show_samples=show_samples,
    )
    return fig, datasets
