"""Declarative study configuration schema and loader."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from motco.simulations.evaluation import SimulationEvaluationParams
from motco.simulations.intersim import InterSIMParams
from motco.simulations.semisynthetic import SemiSyntheticTrajectoryParams

_TRAJECTORY_MODES = {"none", "translation", "magnitude", "orientation", "shape"}
_AXIS_NAMESPACES = {"intersim", "generator", "evaluation"}
_TARGET_KINDS = {"type_i_control", "power_monotonicity", "specificity"}


class StudyConfigError(ValueError):
    """Raised when a study configuration is invalid."""


@dataclass(frozen=True)
class TypeIControlTarget:
    """Type I control target evaluated on null cells."""

    alpha: float
    se_tolerance: float = 2.0
    name: str = "type_i_control"
    kind: str = field(default="type_i_control", init=False)


@dataclass(frozen=True)
class PowerMonotonicityTarget:
    """Monotone power and floor at the largest effect size."""

    trajectory_mode: str
    statistic: str
    min_power_at_top: float
    name: str = "power_monotonicity"
    kind: str = field(default="power_monotonicity", init=False)


@dataclass(frozen=True)
class SpecificityTarget:
    """Off-diagonal specificity (rate ≈ alpha within tolerance)."""

    trajectory_mode: str
    statistic: str
    alpha: float
    se_tolerance: float = 2.0
    name: str = "specificity"
    kind: str = field(default="specificity", init=False)


@dataclass(frozen=True)
class AcceptanceTargets:
    """Collection of pre-specified acceptance targets."""

    type_i: tuple[TypeIControlTarget, ...] = ()
    power: tuple[PowerMonotonicityTarget, ...] = ()
    specificity: tuple[SpecificityTarget, ...] = ()


@dataclass(frozen=True)
class StudyConfig:
    """Declarative study definition."""

    intersim: InterSIMParams
    generator: SemiSyntheticTrajectoryParams
    evaluation: SimulationEvaluationParams
    trajectory_modes: tuple[str, ...]
    effect_sizes: tuple[float, ...]
    axes: Mapping[str, tuple[Any, ...]] = field(default_factory=dict)
    n_replicates: int = 1
    base_seed: int = 0
    alpha: float = 0.05
    acceptance: AcceptanceTargets = field(default_factory=AcceptanceTargets)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_replicates < 0:
            raise StudyConfigError("n_replicates must be non-negative.")
        if not self.trajectory_modes:
            raise StudyConfigError("trajectory_modes must contain at least one mode.")
        unknown = sorted(set(self.trajectory_modes) - _TRAJECTORY_MODES)
        if unknown:
            raise StudyConfigError(f"Unknown trajectory mode(s): {unknown}.")
        if not self.effect_sizes:
            raise StudyConfigError("effect_sizes must contain at least one value.")
        for value in self.effect_sizes:
            if float(value) < 0:
                raise StudyConfigError(f"effect_sizes must be non-negative; got {value}.")
        for axis in self.axes:
            _validate_axis_namespace(axis)
        if not (0 < self.alpha < 1):
            raise StudyConfigError("alpha must be between 0 and 1.")


def load_study_config(path: str | Path) -> StudyConfig:
    """Load a study configuration from a YAML or JSON file."""

    path = Path(path)
    if not path.exists():
        raise StudyConfigError(f"Study configuration not found: {path}.")
    text = path.read_text(encoding="utf-8")
    data = _parse_text(text, path.suffix.lower())
    if not isinstance(data, Mapping):
        raise StudyConfigError(f"Study configuration root must be a mapping, got {type(data).__name__}.")
    return _build_config(data)


def dump_study_config(config: StudyConfig, path: str | Path) -> None:
    """Write a study configuration to JSON (always JSON for portability)."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _config_to_dict(config)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_text(text: str, suffix: str) -> Any:
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore[import-untyped,unused-ignore]
        except ImportError as exc:  # pragma: no cover - exercised only without PyYAML
            raise StudyConfigError(
                "PyYAML is required to load YAML study configurations. "
                "Install pyyaml or rewrite the config as JSON."
            ) from exc
        return yaml.safe_load(text)
    if suffix == ".json":
        return json.loads(text)
    # try YAML first, fall back to JSON
    try:
        import yaml  # type: ignore[import-untyped,unused-ignore]

        return yaml.safe_load(text)
    except ImportError:
        return json.loads(text)


def _build_config(data: Mapping[str, Any]) -> StudyConfig:
    required = {"intersim", "generator", "evaluation", "trajectory_modes", "effect_sizes"}
    missing = sorted(required - set(data))
    if missing:
        raise StudyConfigError(f"Study configuration is missing required field(s): {missing}.")

    intersim = _build_intersim(data["intersim"])
    generator = _build_generator(data["generator"])
    evaluation = _build_evaluation(data.get("evaluation") or {})
    trajectory_modes = tuple(str(mode) for mode in data["trajectory_modes"])
    effect_sizes = tuple(float(value) for value in data["effect_sizes"])
    axes = _build_axes(data.get("axes") or {})
    n_replicates = int(data.get("n_replicates", 1))
    base_seed = int(data.get("base_seed", 0))
    alpha = float(data.get("alpha", 0.05))
    acceptance = _build_acceptance(data.get("acceptance") or {})
    metadata = dict(data.get("metadata") or {})
    return StudyConfig(
        intersim=intersim,
        generator=generator,
        evaluation=evaluation,
        trajectory_modes=trajectory_modes,
        effect_sizes=effect_sizes,
        axes=axes,
        n_replicates=n_replicates,
        base_seed=base_seed,
        alpha=alpha,
        acceptance=acceptance,
        metadata=metadata,
    )


def _build_intersim(raw: Mapping[str, Any]) -> InterSIMParams:
    if "seed" not in raw:
        raise StudyConfigError("intersim.seed is required.")
    field_names = {f.name for f in dataclasses.fields(InterSIMParams)}
    unknown = sorted(set(raw) - field_names)
    if unknown:
        raise StudyConfigError(f"intersim has unknown field(s): {unknown}.")
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(InterSIMParams):
        if f.name not in raw:
            continue
        value = raw[f.name]
        if f.name == "cluster_sample_prop" and value is not None:
            value = tuple(float(v) for v in value)
        kwargs[f.name] = value
    return InterSIMParams(**kwargs)


def _build_generator(raw: Mapping[str, Any]) -> SemiSyntheticTrajectoryParams:
    if "seed" not in raw:
        raise StudyConfigError("generator.seed is required.")
    field_names = {f.name for f in dataclasses.fields(SemiSyntheticTrajectoryParams)}
    unknown = sorted(set(raw) - field_names)
    if unknown:
        raise StudyConfigError(f"generator has unknown field(s): {unknown}.")
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(SemiSyntheticTrajectoryParams):
        if f.name not in raw:
            continue
        value = raw[f.name]
        if f.name == "group_labels" and value is not None:
            value = tuple(str(v) for v in value)
        kwargs[f.name] = value
    return SemiSyntheticTrajectoryParams(**kwargs)


def _build_evaluation(raw: Mapping[str, Any]) -> SimulationEvaluationParams:
    field_names = {f.name for f in dataclasses.fields(SimulationEvaluationParams)}
    unknown = sorted(set(raw) - field_names)
    if unknown:
        raise StudyConfigError(f"evaluation has unknown field(s): {unknown}.")
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(SimulationEvaluationParams):
        if f.name not in raw:
            continue
        kwargs[f.name] = raw[f.name]
    return SimulationEvaluationParams(**kwargs)


def _build_axes(raw: Mapping[str, Any]) -> Mapping[str, tuple[Any, ...]]:
    axes: dict[str, tuple[Any, ...]] = {}
    for axis, values in raw.items():
        _validate_axis_namespace(axis)
        if not isinstance(values, Sequence) or isinstance(values, str | bytes):
            raise StudyConfigError(f"axis {axis!r} values must be a sequence.")
        axes[axis] = tuple(values)
    return axes


def _build_acceptance(raw: Mapping[str, Any]) -> AcceptanceTargets:
    unknown = sorted(set(raw) - {"type_i", "power", "specificity"})
    if unknown:
        raise StudyConfigError(f"acceptance has unknown block(s): {unknown}.")
    type_i = tuple(_build_type_i_target(entry) for entry in raw.get("type_i", []) or [])
    power = tuple(_build_power_target(entry) for entry in raw.get("power", []) or [])
    specificity = tuple(_build_specificity_target(entry) for entry in raw.get("specificity", []) or [])
    return AcceptanceTargets(type_i=type_i, power=power, specificity=specificity)


def _build_type_i_target(raw: Mapping[str, Any]) -> TypeIControlTarget:
    if "alpha" not in raw:
        raise StudyConfigError("acceptance.type_i entries require 'alpha'.")
    return TypeIControlTarget(
        alpha=float(raw["alpha"]),
        se_tolerance=float(raw.get("se_tolerance", 2.0)),
        name=str(raw.get("name", "type_i_control")),
    )


def _build_power_target(raw: Mapping[str, Any]) -> PowerMonotonicityTarget:
    required = {"trajectory_mode", "statistic", "min_power_at_top"}
    missing = sorted(required - set(raw))
    if missing:
        raise StudyConfigError(f"acceptance.power entry missing field(s): {missing}.")
    mode = str(raw["trajectory_mode"])
    if mode not in _TRAJECTORY_MODES:
        raise StudyConfigError(f"acceptance.power trajectory_mode {mode!r} is unknown.")
    return PowerMonotonicityTarget(
        trajectory_mode=mode,
        statistic=str(raw["statistic"]),
        min_power_at_top=float(raw["min_power_at_top"]),
        name=str(raw.get("name", f"power[{mode},{raw['statistic']}]")),
    )


def _build_specificity_target(raw: Mapping[str, Any]) -> SpecificityTarget:
    required = {"trajectory_mode", "statistic", "alpha"}
    missing = sorted(required - set(raw))
    if missing:
        raise StudyConfigError(f"acceptance.specificity entry missing field(s): {missing}.")
    mode = str(raw["trajectory_mode"])
    if mode not in _TRAJECTORY_MODES:
        raise StudyConfigError(f"acceptance.specificity trajectory_mode {mode!r} is unknown.")
    return SpecificityTarget(
        trajectory_mode=mode,
        statistic=str(raw["statistic"]),
        alpha=float(raw["alpha"]),
        se_tolerance=float(raw.get("se_tolerance", 2.0)),
        name=str(raw.get("name", f"specificity[{mode},{raw['statistic']}]")),
    )


def _validate_axis_namespace(axis: str) -> None:
    if "." not in axis:
        raise StudyConfigError(
            f"axis {axis!r} must use a namespace prefix: 'intersim.', 'generator.', or 'evaluation.'."
        )
    namespace, _, field_name = axis.partition(".")
    if namespace not in _AXIS_NAMESPACES:
        raise StudyConfigError(f"axis {axis!r} has unsupported namespace {namespace!r}.")
    if not field_name:
        raise StudyConfigError(f"axis {axis!r} is missing a field name.")


def _config_to_dict(config: StudyConfig) -> dict[str, Any]:
    return {
        "intersim": _dataclass_dict(config.intersim),
        "generator": _dataclass_dict(config.generator),
        "evaluation": _dataclass_dict(config.evaluation),
        "trajectory_modes": list(config.trajectory_modes),
        "effect_sizes": list(config.effect_sizes),
        "axes": {axis: list(values) for axis, values in config.axes.items()},
        "n_replicates": config.n_replicates,
        "base_seed": config.base_seed,
        "alpha": config.alpha,
        "acceptance": {
            "type_i": [_dataclass_dict(t) for t in config.acceptance.type_i],
            "power": [_dataclass_dict(t) for t in config.acceptance.power],
            "specificity": [_dataclass_dict(t) for t in config.acceptance.specificity],
        },
        "metadata": dict(config.metadata),
    }


def _dataclass_dict(value: Any) -> dict[str, Any]:
    return {f.name: _to_jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, tuple | list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


__all__ = [
    "AcceptanceTargets",
    "PowerMonotonicityTarget",
    "SpecificityTarget",
    "StudyConfig",
    "StudyConfigError",
    "TypeIControlTarget",
    "dump_study_config",
    "load_study_config",
]
