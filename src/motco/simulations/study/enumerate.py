"""Combine the Type I and power grids into the full study grid."""

from __future__ import annotations

import dataclasses
from dataclasses import replace
from typing import Any

from motco.simulations.evaluation import SimulationEvaluationParams
from motco.simulations.grid import (
    SimulationCell,
    SimulationGrid,
    enumerate_power_grid,
    enumerate_type_i_grid,
    make_simulation_cell,
)
from motco.simulations.reference import IntersimReference, load_reference
from motco.simulations.semisynthetic import expected_surgery_headroom
from motco.simulations.study.config import StudyConfig, StudyConfigError

# trajectory modes treated as Type I negative controls
NEGATIVE_CONTROL_MODES: tuple[str, ...] = ("none", "translation")

#: Metadata key naming the seed family a cell draws its generator seed from.
SEED_FAMILY_KEY = "seed_family"


def enumerate_study(config: StudyConfig) -> SimulationGrid:
    """Build the combined Type I + power grid for a study."""

    resolver = _attribution_resolver(config)
    type_i = enumerate_type_i_grid(
        baseline_generator_params=config.generator,
        evaluation_params=config.evaluation,
        axes=config.axes,
        n_replicates=config.n_replicates,
        base_seed=config.base_seed,
    )
    cells = list(type_i.cells)
    cells.extend(_negative_control_cells(config))
    cells.extend(_power_cells(config, resolver))

    if config.matched_seeds.enabled:
        cells = _assign_seed_families(cells, config)

    _require_unique_ids(cells)
    _require_surgery_headroom(cells)
    grid_metadata: dict[str, Any] = {
        "grid_type": "study",
        "study_config_metadata": dict(config.metadata),
        "matched_seed_policy": {
            "enabled": config.matched_seeds.enabled,
            "version": config.matched_seeds.version,
            "primary_family": config.matched_seeds.primary_family,
            "shared_zero_effect_anchor": config.matched_seeds.shared_zero_effect_anchor,
        },
    }
    return SimulationGrid(cells=tuple(cells), metadata=grid_metadata)


def _attribution_resolver(config: StudyConfig):
    """Resolve the study attribution selector into per-cell evaluation params."""

    selector = config.attribution
    if not selector.enabled:
        return None
    enabled_params = replace(config.evaluation, attribution=selector.settings())

    def resolve(phase: str, trajectory_mode: str, effect_size: float) -> SimulationEvaluationParams:
        if selector.selects(phase=phase, trajectory_mode=trajectory_mode, effect_size=effect_size):
            return enabled_params
        return config.evaluation

    return resolve


def _power_cells(config: StudyConfig, resolver) -> list[SimulationCell]:
    """Primary and OFAT power cells, with the shared zero-effect anchor when enabled."""

    if not _uses_shared_anchor(config):
        power = enumerate_power_grid(
            baseline_generator_params=config.generator,
            evaluation_params=config.evaluation,
            trajectory_modes=config.trajectory_modes,
            effect_sizes=config.effect_sizes,
            axes=config.axes,
            n_replicates=config.n_replicates,
            base_seed=config.base_seed,
            evaluation_resolver=resolver,
        )
        return list(power.cells)

    nonzero_effects = tuple(value for value in config.effect_sizes if float(value) != 0.0)
    if not nonzero_effects:
        raise StudyConfigError(
            "matched_seeds.shared_zero_effect_anchor requires at least one nonzero effect size."
        )
    power = enumerate_power_grid(
        baseline_generator_params=config.generator,
        evaluation_params=config.evaluation,
        trajectory_modes=config.trajectory_modes,
        effect_sizes=nonzero_effects,
        axes=config.axes,
        n_replicates=config.n_replicates,
        base_seed=config.base_seed,
        evaluation_resolver=resolver,
    )
    return [_zero_effect_anchor_cell(config), *power.cells]


def _zero_effect_anchor_cell(config: StudyConfig) -> SimulationCell:
    """One mode-agnostic zero-effect primary cell shared by every mode's curve.

    At ``group_effect_size == 0`` the generator returns group B's baseline
    unchanged and consumes no extra randomness, so per-mode zero-effect cells
    inside one matched-seed family would be byte-identical datasets. A single
    anchor is enumerated instead and every mode's ``0.00`` power point resolves
    to it.
    """

    generator = replace(
        config.generator,
        trajectory_mode="none",
        group_effect_size=0.0,
    )
    return make_simulation_cell(
        phase="power_primary",
        generator_params=generator,
        evaluation_params=config.evaluation,
        n_replicates=config.n_replicates,
        base_seed=config.base_seed,
        metadata={
            "trajectory_mode": "none",
            "effect_size": 0.0,
            "varied_axis": None,
            "zero_effect_anchor": True,
            "resolves_modes": list(config.trajectory_modes),
        },
    )


def _negative_control_cells(config: StudyConfig) -> list[SimulationCell]:
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


def _assign_seed_families(cells: list[SimulationCell], config: StudyConfig) -> list[SimulationCell]:
    """Stamp each cell with the seed family its generator seed is drawn from.

    Primary power cells (including the zero-effect anchor) share one family so
    their datasets are paired at the same replicate index. Every other cell —
    Type I baselines, negative controls, and OFAT cells — keeps a family of its
    own so it remains an independent draw.
    """

    family = str(config.matched_seeds.primary_family)
    out: list[SimulationCell] = []
    for cell in cells:
        assigned = family if cell.phase == "power_primary" else f"control:{cell.cell_id}"
        out.append(replace(cell, metadata={**dict(cell.metadata), SEED_FAMILY_KEY: assigned}))
    return out


def _uses_shared_anchor(config: StudyConfig) -> bool:
    return (
        config.matched_seeds.enabled
        and config.matched_seeds.shared_zero_effect_anchor
        and any(float(value) == 0.0 for value in config.effect_sizes)
    )


def _require_unique_ids(cells: list[SimulationCell]) -> None:
    ids = [cell.cell_id for cell in cells]
    duplicates = sorted({cid for cid in ids if ids.count(cid) > 1})
    if duplicates:
        raise StudyConfigError(f"Study enumeration produced duplicate cell_id(s): {duplicates}.")
    if len(cells) < 2:
        return
    _require_distinct_primary_datasets(cells)


def _require_surgery_headroom(cells: list[SimulationCell]) -> None:
    """Reject cells whose requested effect exceeds the expected surgery headroom.

    A pool-limited surgery (orientation, translation, shape/relocate) that
    cannot be realized in full turns distinct requested effects into the same
    realized construction. The runtime policy catches it per replicate; this
    catches it at configuration time, before any cluster compute is spent.

    Cells that explicitly opt into ``surgery_censoring="clamp"`` are exempt —
    they have accepted partial surgeries, and their records carry the censored
    flag for the summaries to surface.

    The headroom is read from each cell's *own* generator parameters, so a study
    sweeping ``generator.baseline_continuity`` is checked per continuity value:
    higher continuity shrinks the stage-active union and grows the destination
    pool, so an effect that fails at ρ = 0 may enumerate at a higher ρ.
    """

    reference: IntersimReference | None = None
    offenders: list[str] = []
    for cell in cells:
        params = cell.generator_params
        if getattr(params, "surgery_censoring", "error") != "error":
            continue
        if reference is None:
            reference = load_reference()
        headroom = expected_surgery_headroom(params, reference=reference)
        if headroom is None or headroom.fits:
            continue
        offenders.append(
            f"  - cell {cell.cell_id!r}: trajectory_mode={headroom.trajectory_mode!r} at "
            f"group_effect_size={headroom.group_effect_size:g} requests ~{headroom.nominal:.0f} "
            f"site(s), but the expected destination pool is ~{headroom.pool:.0f} less a "
            f"{headroom.guard_band:.0f}-site guard band (~{headroom.available:.0f} available); "
            f"it saturates at group_effect_size≈{headroom.saturating_effect:.2f}"
        )
    if not offenders:
        return
    raise StudyConfigError(
        f"{len(offenders)} cell(s) request more surgery than the expected pool headroom, so "
        "their realized constructions would be censored and near-identical across effects "
        "instead of independent measurements:\n"
        + "\n".join(offenders)
        + "\nLower the requested effects (or p_dmp), or set generator.surgery_censoring='clamp' "
        "to accept partial surgeries."
    )


def _require_distinct_primary_datasets(cells: list[SimulationCell]) -> None:
    """No two primary cells in one seed family may generate identical datasets.

    Within a matched-seed family every primary cell draws the same generator
    seed at a given replicate index, so two cells whose generator parameters are
    otherwise identical would produce byte-identical data and report duplicated
    evidence as if it were independent.
    """

    seen: dict[tuple[str, tuple[Any, ...]], str] = {}
    for cell in cells:
        if cell.phase != "power_primary":
            continue
        family = str(cell.metadata.get(SEED_FAMILY_KEY, cell.cell_id))
        key = (family, _generator_identity(cell))
        previous = seen.get(key)
        if previous is not None:
            raise StudyConfigError(
                f"Cells {previous!r} and {cell.cell_id!r} share seed family {family!r} and identical "
                "generator parameters, so they would generate identical datasets at every replicate index."
            )
        seen[key] = cell.cell_id


#: Generator fields the group-B transform ignores once the requested effect is zero.
_MODE_ONLY_FIELDS: tuple[str, ...] = ("trajectory_mode", "shape_kind", "magnitude_kind")


def _generator_identity(cell: SimulationCell) -> tuple[Any, ...]:
    """Generator parameters that determine the sampled dataset, seed excluded.

    At ``group_effect_size == 0`` the group-B transform short-circuits to group
    A's baseline, so the mode-selecting fields no longer distinguish datasets and
    are normalized away — that is precisely the collision the shared zero-effect
    anchor exists to prevent.
    """

    params = cell.generator_params
    is_null = float(getattr(params, "group_effect_size", 0.0)) == 0.0
    values: list[tuple[str, Any]] = []
    for f in sorted(dataclasses.fields(params), key=lambda f: f.name):
        if f.name == "seed":
            continue
        if is_null and f.name in _MODE_ONLY_FIELDS:
            continue
        values.append((f.name, _hashable(getattr(params, f.name))))
    if is_null:
        values.append(("trajectory_mode", "none"))
    return tuple(values)


def _hashable(value: Any) -> Any:
    if isinstance(value, list | tuple):
        return tuple(_hashable(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((str(key), _hashable(item)) for key, item in value.items()))
    return value


__all__ = ["NEGATIVE_CONTROL_MODES", "SEED_FAMILY_KEY", "enumerate_study"]
