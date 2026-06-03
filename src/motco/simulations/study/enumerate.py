"""Combine the Type I and power grids into the full study grid."""

from __future__ import annotations

from dataclasses import replace

from motco.simulations.grid import (
    SimulationGrid,
    enumerate_power_grid,
    enumerate_type_i_grid,
    make_simulation_cell,
)
from motco.simulations.study.config import StudyConfig, StudyConfigError

# trajectory modes treated as Type I negative controls
NEGATIVE_CONTROL_MODES: tuple[str, ...] = ("none", "translation")


def enumerate_study(config: StudyConfig) -> SimulationGrid:
    """Build the combined Type I + power grid for a study."""

    type_i = enumerate_type_i_grid(
        baseline_generator_params=config.generator,
        evaluation_params=config.evaluation,
        axes=config.axes,
        n_replicates=config.n_replicates,
        base_seed=config.base_seed,
    )
    power = enumerate_power_grid(
        baseline_generator_params=config.generator,
        evaluation_params=config.evaluation,
        trajectory_modes=config.trajectory_modes,
        effect_sizes=config.effect_sizes,
        axes=config.axes,
        n_replicates=config.n_replicates,
        base_seed=config.base_seed,
    )
    cells = list(type_i.cells)
    cells.extend(_negative_control_cells(config))
    cells.extend(power.cells)

    _require_unique_ids(cells)
    return SimulationGrid(
        cells=tuple(cells),
        metadata={"grid_type": "study", "study_config_metadata": dict(config.metadata)},
    )


def _negative_control_cells(config: StudyConfig) -> list:
    """Ensure the `none` and `translation` modes appear as Type I negative controls.

    The Type I grid built by ``enumerate_type_i_grid`` covers the ``none`` baseline
    (group_effect_size = 0). For the study, an explicit ``translation`` Type I cell is
    also required so the deliverable matrix has both negative-control rows.
    """

    cells = []
    for mode in NEGATIVE_CONTROL_MODES:
        if mode == "none":
            # already covered by enumerate_type_i_grid's baseline cell
            continue
        generator = replace(
            config.generator,
            trajectory_mode=mode,  # type: ignore[arg-type]
            group_effect_size=float(config.effect_sizes[-1]),
        )
        cells.append(
            make_simulation_cell(
                phase="type_i_baseline",
                generator_params=generator,
                evaluation_params=config.evaluation,
                n_replicates=config.n_replicates,
                base_seed=config.base_seed,
                metadata={
                    "varied_axis": None,
                    "varied_value": None,
                    "trajectory_mode": mode,
                    "effect_size": float(config.effect_sizes[-1]),
                    "negative_control": True,
                },
            )
        )
    return cells


def _require_unique_ids(cells: list) -> None:
    ids = [cell.cell_id for cell in cells]
    duplicates = sorted({cid for cid in ids if ids.count(cid) > 1})
    if duplicates:
        raise StudyConfigError(f"Study enumeration produced duplicate cell_id(s): {duplicates}.")


__all__ = ["enumerate_study", "NEGATIVE_CONTROL_MODES"]
