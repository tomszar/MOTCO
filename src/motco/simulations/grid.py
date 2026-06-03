"""Simulation grid orchestration for semi-synthetic trajectory studies."""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from motco.simulations.evaluation import SimulationEvaluationParams, SimulationEvaluationResult
from motco.simulations.semisynthetic import (
    SemiSyntheticTrajectoryParams,
    generate_semisynthetic_trajectory,
)

SimulationPhase = Literal[
    "type_i_baseline",
    "type_i_ofat",
    "power_primary",
    "power_ofat",
]
ErrorPolicy = Literal["raise", "record"]
PersistenceFormat = Literal["jsonl"]


class SimulationGridError(ValueError):
    """Raised when simulation grid orchestration inputs are invalid."""


@dataclass(frozen=True)
class SimulationCell:
    """One simulation parameter cell with one or more replicates."""

    cell_id: str
    phase: str
    generator_params: SemiSyntheticTrajectoryParams
    evaluation_params: SimulationEvaluationParams
    n_replicates: int = 1
    base_seed: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_replicates < 1:
            raise SimulationGridError("n_replicates must be at least 1.")
        if not self.cell_id:
            raise SimulationGridError("cell_id must be a non-empty string.")
        if not self.phase:
            raise SimulationGridError("phase must be a non-empty string.")


@dataclass(frozen=True)
class SimulationGrid:
    """Collection of simulation cells."""

    cells: tuple[SimulationCell, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ids = [cell.cell_id for cell in self.cells]
        duplicates = sorted({cell_id for cell_id in ids if ids.count(cell_id) > 1})
        if duplicates:
            raise SimulationGridError(f"cell_id values must be unique; duplicates: {duplicates}.")


@dataclass(frozen=True)
class SimulationReplicateResult:
    """One persisted row for a simulation cell replicate."""

    cell_id: str
    phase: str
    replicate_index: int
    replicate_seed: int
    generator_seed: int
    evaluation_seed: int | None
    parameter_signature: str
    status: Literal["completed", "failed"]
    p_values: dict[str, float | None] = field(default_factory=dict)
    pair_statistics: dict[str, float | None] = field(default_factory=dict)
    truth_metadata: dict[str, Any] = field(default_factory=dict)
    runtime_metadata: dict[str, Any] = field(default_factory=dict)
    cell_metadata: dict[str, Any] = field(default_factory=dict)
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class SimulationSummaryResult:
    """Rejection-rate summary for one cell and statistic."""

    cell_id: str
    phase: str
    statistic: str
    alpha: float
    completed_replicates: int
    available_replicates: int
    rejected_replicates: int
    rejection_rate: float | None
    monte_carlo_se: float | None
    unavailable_replicates: int


@dataclass(frozen=True)
class SimulationRunConfig:
    """Runtime options for local grid execution."""

    output_path: Path | None = None
    persistence_format: PersistenceFormat = "jsonl"
    resume: bool = True
    overwrite: bool = False
    error_policy: ErrorPolicy = "raise"


Evaluator = Callable[
    [SemiSyntheticTrajectoryParams, SimulationEvaluationParams],
    SimulationEvaluationResult,
]


def make_simulation_cell(
    *,
    phase: str,
    generator_params: SemiSyntheticTrajectoryParams,
    evaluation_params: SimulationEvaluationParams | None = None,
    n_replicates: int = 1,
    base_seed: int = 0,
    metadata: Mapping[str, Any] | None = None,
    cell_id: str | None = None,
) -> SimulationCell:
    """Create a simulation cell, generating a stable ID when one is not supplied."""

    evaluation_params = evaluation_params or SimulationEvaluationParams()
    payload = {
        "phase": phase,
        "generator_params": _to_jsonable(generator_params),
        "evaluation_params": _to_jsonable(evaluation_params),
        "n_replicates": n_replicates,
        "base_seed": base_seed,
        "metadata": _to_jsonable(metadata or {}),
    }
    resolved_id = cell_id or f"{phase}-{_stable_digest(payload, length=12)}"
    return SimulationCell(
        cell_id=resolved_id,
        phase=phase,
        generator_params=generator_params,
        evaluation_params=evaluation_params,
        n_replicates=n_replicates,
        base_seed=base_seed,
        metadata=dict(metadata or {}),
    )


def enumerate_type_i_grid(
    *,
    baseline_generator_params: SemiSyntheticTrajectoryParams,
    evaluation_params: SimulationEvaluationParams | None = None,
    axes: Mapping[str, Sequence[Any]] | None = None,
    n_replicates: int = 1,
    base_seed: int = 0,
) -> SimulationGrid:
    """Enumerate a null Type I grid with baseline plus one-factor-at-a-time axes."""

    null_generator = replace(baseline_generator_params, trajectory_mode="none", group_effect_size=0.0)
    cells = [
        make_simulation_cell(
            phase="type_i_baseline",
            generator_params=null_generator,
            evaluation_params=evaluation_params,
            n_replicates=n_replicates,
            base_seed=base_seed,
            metadata={"varied_axis": None, "varied_value": None},
        )
    ]
    for axis, values in (axes or {}).items():
        baseline_value = _get_axis_value(null_generator, evaluation_params, axis)
        for value in values:
            if _to_jsonable(value) == _to_jsonable(baseline_value):
                continue
            generator_params, eval_params = _apply_axis_value(
                null_generator,
                evaluation_params or SimulationEvaluationParams(),
                axis,
                value,
            )
            cells.append(
                make_simulation_cell(
                    phase="type_i_ofat",
                    generator_params=generator_params,
                    evaluation_params=eval_params,
                    n_replicates=n_replicates,
                    base_seed=base_seed,
                    metadata={"varied_axis": axis, "varied_value": value},
                )
            )
    return SimulationGrid(cells=tuple(cells), metadata={"grid_type": "type_i"})


def enumerate_power_grid(
    *,
    baseline_generator_params: SemiSyntheticTrajectoryParams,
    evaluation_params: SimulationEvaluationParams | None = None,
    trajectory_modes: Sequence[str],
    effect_sizes: Sequence[float],
    axes: Mapping[str, Sequence[Any]] | None = None,
    n_replicates: int = 1,
    base_seed: int = 0,
) -> SimulationGrid:
    """Enumerate power cells from trajectory modes, effect sizes, and optional axes."""

    if not trajectory_modes:
        raise SimulationGridError("trajectory_modes must contain at least one mode.")
    if not effect_sizes:
        raise SimulationGridError("effect_sizes must contain at least one value.")
    cells: list[SimulationCell] = []
    eval_params = evaluation_params or SimulationEvaluationParams()
    for mode, effect_size in itertools.product(trajectory_modes, effect_sizes):
        generator = replace(
            baseline_generator_params,
            trajectory_mode=mode,  # type: ignore[arg-type]
            group_effect_size=float(effect_size),
        )
        cells.append(
            make_simulation_cell(
                phase="power_primary",
                generator_params=generator,
                evaluation_params=eval_params,
                n_replicates=n_replicates,
                base_seed=base_seed,
                metadata={"trajectory_mode": mode, "effect_size": float(effect_size), "varied_axis": None},
            )
        )
        for axis, values in (axes or {}).items():
            baseline_value = _get_axis_value(generator, eval_params, axis)
            for value in values:
                if _to_jsonable(value) == _to_jsonable(baseline_value):
                    continue
                generator_params, axis_eval_params = _apply_axis_value(
                    generator,
                    eval_params,
                    axis,
                    value,
                )
                cells.append(
                    make_simulation_cell(
                        phase="power_ofat",
                        generator_params=generator_params,
                        evaluation_params=axis_eval_params,
                        n_replicates=n_replicates,
                        base_seed=base_seed,
                        metadata={
                            "trajectory_mode": mode,
                            "effect_size": float(effect_size),
                            "varied_axis": axis,
                            "varied_value": value,
                        },
                    )
                )
    return SimulationGrid(cells=tuple(cells), metadata={"grid_type": "power"})


def derive_replicate_seed(cell: SimulationCell, replicate_index: int) -> int:
    """Derive a deterministic 31-bit unsigned seed from cell identity and replicate index.

    The result is masked into ``[0, 2**31 - 1]`` so it seeds numpy's default RNG
    (and any other downstream RNG) reproducibly.
    """

    if replicate_index < 0 or replicate_index >= cell.n_replicates:
        raise SimulationGridError(
            f"replicate_index must be in [0, {cell.n_replicates - 1}], got {replicate_index}."
        )
    payload = {"base_seed": cell.base_seed, "cell_id": cell.cell_id, "replicate_index": replicate_index}
    return int(_stable_digest(payload, length=8), 16) & 0x7FFFFFFF


def parameter_signature(cell: SimulationCell) -> str:
    """Return a stable hash of all parameters that define a cell."""

    payload = {
        "phase": cell.phase,
        "generator_params": _to_jsonable(cell.generator_params),
        "evaluation_params": _to_jsonable(cell.evaluation_params),
        "n_replicates": cell.n_replicates,
        "base_seed": cell.base_seed,
        "metadata": _to_jsonable(cell.metadata),
        "seed_derivation_version": 3,
    }
    return _stable_digest(payload)


def run_simulation_replicate(
    cell: SimulationCell,
    replicate_index: int,
    *,
    evaluator: Evaluator | None = None,
    error_policy: ErrorPolicy = "raise",
) -> SimulationReplicateResult:
    """Run one replicate, using a fake evaluator in tests or the default harness in production."""

    _validate_error_policy(error_policy)
    replicate_seed = derive_replicate_seed(cell, replicate_index)
    generator_params = replace(cell.generator_params, seed=replicate_seed)
    evaluation_seed = cell.evaluation_params.seed if cell.evaluation_params.seed is not None else replicate_seed
    evaluation_params = replace(cell.evaluation_params, seed=evaluation_seed)
    evaluator = evaluator or _default_evaluator
    start = perf_counter()
    try:
        result = evaluator(generator_params, evaluation_params)
    except Exception as exc:
        if error_policy == "raise":
            raise
        return SimulationReplicateResult(
            cell_id=cell.cell_id,
            phase=cell.phase,
            replicate_index=replicate_index,
            replicate_seed=replicate_seed,
            generator_seed=generator_params.seed,
            evaluation_seed=evaluation_params.seed,
            parameter_signature=parameter_signature(cell),
            status="failed",
            runtime_metadata={"runtime_seconds": perf_counter() - start},
            cell_metadata=dict(cell.metadata),
            error_type=type(exc).__name__,
            error_message=str(exc),
        )

    return SimulationReplicateResult(
        cell_id=cell.cell_id,
        phase=cell.phase,
        replicate_index=replicate_index,
        replicate_seed=replicate_seed,
        generator_seed=generator_params.seed,
        evaluation_seed=evaluation_params.seed,
        parameter_signature=parameter_signature(cell),
        status="completed",
        p_values=_finite_mapping(result.p_values),
        pair_statistics=_finite_mapping(result.pair_statistics),
        truth_metadata=result.truth_metadata,
        runtime_metadata=result.runtime_metadata,
        cell_metadata=dict(cell.metadata),
    )


def run_simulation_grid(
    grid: SimulationGrid,
    *,
    config: SimulationRunConfig | None = None,
    evaluator: Evaluator | None = None,
) -> list[SimulationReplicateResult]:
    """Run a grid locally with optional JSONL persistence and resume support."""

    config = config or SimulationRunConfig()
    if config.persistence_format != "jsonl":
        raise SimulationGridError("Only JSONL persistence is currently supported.")
    completed = (
        _completed_index(config.output_path)
        if config.output_path and config.resume and not config.overwrite
        else {}
    )
    records: list[SimulationReplicateResult] = []
    for cell in grid.cells:
        signature = parameter_signature(cell)
        for replicate_index in range(cell.n_replicates):
            key = (cell.cell_id, replicate_index)
            existing_signature = completed.get(key)
            if existing_signature is not None:
                if existing_signature != signature:
                    if not config.overwrite:
                        raise SimulationGridError(
                            f"Existing result for {cell.cell_id} replicate {replicate_index} has a different "
                            "parameter signature. Set overwrite=True to rerun."
                        )
                else:
                    continue
            record = run_simulation_replicate(
                cell,
                replicate_index,
                evaluator=evaluator,
                error_policy=config.error_policy,
            )
            records.append(record)
            if config.output_path is not None:
                append_replicate_results(config.output_path, [record])
    return records


def append_replicate_results(path: Path, records: Iterable[SimulationReplicateResult]) -> None:
    """Append replicate records to a JSONL file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_to_jsonable(record), sort_keys=True, allow_nan=False) + "\n")


def read_replicate_results(path: Path) -> list[SimulationReplicateResult]:
    """Read replicate records from a JSONL file."""

    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(_replicate_result_from_dict(json.loads(line)))
    return records


def summarize_rejection_rates(
    records: Sequence[SimulationReplicateResult],
    *,
    alpha: float = 0.05,
    statistics: Sequence[str] = ("delta", "angle", "shape"),
) -> list[SimulationSummaryResult]:
    """Summarize p-value rejection rates by cell and statistic."""

    if not (0 < alpha < 1):
        raise SimulationGridError("alpha must be between 0 and 1.")
    groups: dict[tuple[str, str], list[SimulationReplicateResult]] = {}
    for record in records:
        if record.status != "completed":
            continue
        groups.setdefault((record.cell_id, record.phase), []).append(record)

    summaries: list[SimulationSummaryResult] = []
    for (cell_id, phase), group_records in sorted(groups.items()):
        for statistic in statistics:
            p_values = [record.p_values.get(statistic) for record in group_records]
            available = [float(p) for p in p_values if p is not None and math.isfinite(float(p))]
            rejected = sum(p < alpha for p in available)
            rate = rejected / len(available) if available else None
            se = math.sqrt(rate * (1.0 - rate) / len(available)) if rate is not None else None
            summaries.append(
                SimulationSummaryResult(
                    cell_id=cell_id,
                    phase=phase,
                    statistic=statistic,
                    alpha=alpha,
                    completed_replicates=len(group_records),
                    available_replicates=len(available),
                    rejected_replicates=rejected,
                    rejection_rate=rate,
                    monte_carlo_se=se,
                    unavailable_replicates=len(group_records) - len(available),
                )
            )
    return summaries


def rejection_indicator(p_value: float | None, *, alpha: float = 0.05) -> bool | None:
    """Return whether a p-value rejects at alpha, or None when unavailable."""

    if not (0 < alpha < 1):
        raise SimulationGridError("alpha must be between 0 and 1.")
    if p_value is None or not math.isfinite(float(p_value)):
        return None
    return float(p_value) < alpha


def _default_evaluator(
    generator_params: SemiSyntheticTrajectoryParams,
    evaluation_params: SimulationEvaluationParams,
) -> SimulationEvaluationResult:
    from motco.simulations.evaluation import evaluate_semisynthetic_trajectory

    dataset = generate_semisynthetic_trajectory(generator_params)
    return evaluate_semisynthetic_trajectory(dataset, evaluation_params)


def _apply_axis_value(
    generator_params: SemiSyntheticTrajectoryParams,
    evaluation_params: SimulationEvaluationParams,
    axis: str,
    value: Any,
) -> tuple[SemiSyntheticTrajectoryParams, SimulationEvaluationParams]:
    namespace, field_name = _split_axis(axis)
    if namespace == "generator":
        return replace(generator_params, **{field_name: value}), evaluation_params
    if namespace == "evaluation":
        return generator_params, replace(evaluation_params, **{field_name: value})
    raise SimulationGridError(f"Unsupported axis namespace: {namespace!r}.")


def _get_axis_value(
    generator_params: SemiSyntheticTrajectoryParams,
    evaluation_params: SimulationEvaluationParams | None,
    axis: str,
) -> Any:
    namespace, field_name = _split_axis(axis)
    if namespace == "generator":
        return getattr(generator_params, field_name)
    if namespace == "evaluation":
        return getattr(evaluation_params or SimulationEvaluationParams(), field_name)
    raise SimulationGridError(f"Unsupported axis namespace: {namespace!r}.")


def _split_axis(axis: str) -> tuple[str, str]:
    if "." not in axis:
        raise SimulationGridError(
            f"Axis {axis!r} must use a namespace prefix: 'generator.' or 'evaluation.'."
        )
    namespace, field_name = axis.split(".", 1)
    if not field_name:
        raise SimulationGridError(f"Axis {axis!r} is missing a field name.")
    if namespace not in {"generator", "evaluation"}:
        raise SimulationGridError(f"Unsupported axis namespace: {namespace!r}.")
    return namespace, field_name


def _completed_index(path: Path | None) -> dict[tuple[str, int], str]:
    if path is None or not path.exists():
        return {}
    out = {}
    for record in read_replicate_results(path):
        if record.status == "completed":
            out[(record.cell_id, record.replicate_index)] = record.parameter_signature
    return out


def _replicate_result_from_dict(data: Mapping[str, Any]) -> SimulationReplicateResult:
    return SimulationReplicateResult(
        cell_id=str(data["cell_id"]),
        phase=str(data["phase"]),
        replicate_index=int(data["replicate_index"]),
        replicate_seed=int(data["replicate_seed"]),
        generator_seed=int(data["generator_seed"]),
        evaluation_seed=None if data.get("evaluation_seed") is None else int(data["evaluation_seed"]),
        parameter_signature=str(data["parameter_signature"]),
        status=data["status"],
        p_values=dict(data.get("p_values", {})),
        pair_statistics=dict(data.get("pair_statistics", {})),
        truth_metadata=dict(data.get("truth_metadata", {})),
        runtime_metadata=dict(data.get("runtime_metadata", {})),
        cell_metadata=dict(data.get("cell_metadata", {})),
        error_type=data.get("error_type"),
        error_message=data.get("error_message"),
    )


def _finite_mapping(values: Mapping[str, float]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for key, value in values.items():
        out[key] = float(value) if math.isfinite(float(value)) else None
    return out


def _validate_error_policy(error_policy: ErrorPolicy) -> None:
    if error_policy not in {"raise", "record"}:
        raise SimulationGridError(f"Unsupported error_policy: {error_policy!r}.")


def _stable_digest(payload: Any, *, length: int | None = None) -> str:
    encoded = json.dumps(_to_jsonable(payload), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    return digest[:length] if length is not None else digest


def _to_jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {field.name: _to_jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, tuple | list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value
