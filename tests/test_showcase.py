"""Tests for the trajectory-mode showcase (src/motco/simulations/showcase.py).

These run on the numpy generator and cached reference data (no R).
"""

from __future__ import annotations

import matplotlib
import pytest
from matplotlib.figure import Figure

matplotlib.use("Agg")

from motco.simulations.reference import load_reference
from motco.simulations.showcase import (
    TRAJECTORY_SHOWCASE_MODES,
    TrajectoryShowcaseError,
    build_showcase_figure,
    generate_showcase_datasets,
)


@pytest.fixture(scope="module")
def reference():
    return load_reference()


def test_generate_showcase_datasets_covers_all_modes_and_null(reference):
    datasets = generate_showcase_datasets(
        effect_size=2.0, seed=7, n_samples=180, reference=reference
    )

    assert tuple(datasets.keys()) == TRAJECTORY_SHOWCASE_MODES
    # 'none' is always the null: zero effect regardless of effect_size.
    assert datasets["none"].truth["group_effect_size"] == 0.0
    assert datasets["orientation"].truth["group_effect_size"] == 2.0
    # Shared seed → identical stage/group assignment across modes.
    assert datasets["none"].metadata["stage"].tolist() == datasets["shape"].metadata["stage"].tolist()
    assert datasets["none"].metadata["group"].tolist() == datasets["shape"].metadata["group"].tolist()


def test_generate_showcase_datasets_rejects_empty_modes(reference):
    with pytest.raises(TrajectoryShowcaseError):
        generate_showcase_datasets(modes=(), reference=reference)


def test_build_showcase_figure_smoke(reference):
    datasets = generate_showcase_datasets(
        effect_size=1.5, seed=3, n_samples=180, reference=reference
    )
    fig = build_showcase_figure(datasets, show_samples=True)

    assert isinstance(fig, Figure)
    # One visible axis per scenario; the rest of the grid is hidden.
    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible) == len(datasets)
    # PLS axis labels propagate through the projector wiring.
    assert visible[0].get_xlabel().startswith("PLS1")

    import matplotlib.pyplot as plt

    plt.close(fig)


def test_build_showcase_figure_rejects_empty():
    with pytest.raises(TrajectoryShowcaseError):
        build_showcase_figure({})
