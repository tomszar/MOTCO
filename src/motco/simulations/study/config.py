"""Declarative study configuration schema and loader."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from motco.simulations.evaluation import AttributionDiagnosticSettings, SimulationEvaluationParams
from motco.simulations.semisynthetic import SemiSyntheticTrajectoryParams

_TRAJECTORY_MODES = {"none", "translation", "magnitude", "orientation", "shape"}
_AXIS_NAMESPACES = {"generator", "evaluation"}
_TARGET_KINDS = {"type_i_control", "power_monotonicity", "specificity"}
_GATE_ROLES = {"mandatory_power", "mandatory_control", "descriptive"}
_STATISTICS = {"delta", "angle", "shape"}


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
class AttributionSelector:
    """Which enumerated cells receive orientation-attribution diagnostics.

    The selector is resolved into per-cell evaluation parameters during
    enumeration, so eligibility and the effective bootstrap settings are part of
    the cell's parameter signature rather than something inferred after a run.
    Selection is never conditioned on an observed p-value.
    """

    enabled: bool = False
    trajectory_modes: tuple[str, ...] = ("orientation",)
    phases: tuple[str, ...] = ("power_primary",)
    nonzero_effects_only: bool = True
    effect_sizes: tuple[float, ...] | None = None
    bootstrap_replicates: int = 100
    bootstrap_seed: int | None = 0
    top_k: int = 20
    zero_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        unknown = sorted(set(self.trajectory_modes) - _TRAJECTORY_MODES)
        if unknown:
            raise StudyConfigError(f"attribution.trajectory_modes has unknown mode(s): {unknown}.")
        if not self.trajectory_modes:
            raise StudyConfigError("attribution.trajectory_modes must contain at least one mode.")
        if not self.phases:
            raise StudyConfigError("attribution.phases must contain at least one phase.")
        if self.effect_sizes is not None:
            if not self.effect_sizes:
                raise StudyConfigError("attribution.effect_sizes must be null or a non-empty sequence.")
            for value in self.effect_sizes:
                if float(value) < 0:
                    raise StudyConfigError(f"attribution.effect_sizes must be non-negative; got {value}.")
        if self.bootstrap_replicates < 0:
            raise StudyConfigError("attribution.bootstrap_replicates must be non-negative.")
        if self.top_k < 1:
            raise StudyConfigError("attribution.top_k must be at least 1.")
        if not (self.zero_tolerance > 0):
            raise StudyConfigError("attribution.zero_tolerance must be positive.")

    def settings(self) -> AttributionDiagnosticSettings:
        """Effective evaluation-level settings implied by this selector."""

        return AttributionDiagnosticSettings(
            enabled=True,
            bootstrap_replicates=int(self.bootstrap_replicates),
            bootstrap_seed=None if self.bootstrap_seed is None else int(self.bootstrap_seed),
            top_k=int(self.top_k),
            zero_tolerance=float(self.zero_tolerance),
        )

    def selects(self, *, phase: str, trajectory_mode: str | None, effect_size: float | None) -> bool:
        """Whether a cell with this identity is eligible for attribution."""

        if not self.enabled:
            return False
        if phase not in self.phases:
            return False
        if trajectory_mode is None or trajectory_mode not in self.trajectory_modes:
            return False
        if effect_size is None:
            return False
        value = float(effect_size)
        if self.nonzero_effects_only and value == 0.0:
            return False
        if self.effect_sizes is not None:
            return any(abs(value - float(candidate)) < 1e-12 for candidate in self.effect_sizes)
        return True


@dataclass(frozen=True)
class MatchedSeedPolicy:
    """Opt-in matched generator seeds shared across primary cells.

    When enabled, every primary power cell draws its generator seed from a
    shared ``(seed family, replicate index)`` key instead of its own cell id, so
    requested-effect comparisons are paired at the generated-reference level.
    Negative-control and OFAT cells keep their own seed families so they stay
    independent. The policy version enters the parameter signature.
    """

    enabled: bool = False
    version: int = 1
    primary_family: str = "primary"
    shared_zero_effect_anchor: bool = True

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        if self.version < 1:
            raise StudyConfigError("matched_seeds.version must be at least 1.")
        if not str(self.primary_family).strip():
            raise StudyConfigError("matched_seeds.primary_family must be a non-empty string.")


@dataclass(frozen=True)
class GateRule:
    """One predeclared mode/statistic gate role."""

    trajectory_mode: str
    statistic: str
    role: str
    min_power_at_top: float | None = None

    def __post_init__(self) -> None:
        if self.trajectory_mode not in _TRAJECTORY_MODES:
            raise StudyConfigError(f"gate rule trajectory_mode {self.trajectory_mode!r} is unknown.")
        if self.statistic not in _STATISTICS:
            raise StudyConfigError(f"gate rule statistic {self.statistic!r} is unknown.")
        if self.role not in _GATE_ROLES:
            raise StudyConfigError(f"gate rule role {self.role!r} is unknown; expected one of {sorted(_GATE_ROLES)}.")

    @property
    def name(self) -> str:
        return f"{self.role}[{self.trajectory_mode},{self.statistic}]"


@dataclass(frozen=True)
class Phase4GateConfig:
    """Every parameter the Phase 4 gate consumes.

    Summary and report code reads all thresholds from here and hard-codes none
    of its own, so a gate can be re-specified by editing the committed config.
    """

    enabled: bool = False
    alpha: float = 0.05
    control_se_tolerance: float = 2.0
    monotonicity_se_tolerance: float = 2.0
    min_power_at_top: float = 0.80
    confirmation_se_threshold: float = 1.0
    max_marginal_exceedances: int = 1
    control_modes: tuple[str, ...] = ("none", "translation")
    rules: tuple[GateRule, ...] = ()
    require_complete_records: bool = True

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        if not (0 < self.alpha < 1):
            raise StudyConfigError("acceptance.gate.alpha must be between 0 and 1.")
        for name in ("control_se_tolerance", "monotonicity_se_tolerance", "confirmation_se_threshold"):
            value = float(getattr(self, name))
            if value < 0:
                raise StudyConfigError(f"acceptance.gate.{name} must be non-negative.")
        if not (0 <= self.min_power_at_top <= 1):
            raise StudyConfigError("acceptance.gate.min_power_at_top must be between 0 and 1.")
        if self.max_marginal_exceedances < 0:
            raise StudyConfigError("acceptance.gate.max_marginal_exceedances must be non-negative.")
        unknown = sorted(set(self.control_modes) - _TRAJECTORY_MODES)
        if unknown:
            raise StudyConfigError(f"acceptance.gate.control_modes has unknown mode(s): {unknown}.")
        if not self.control_modes:
            raise StudyConfigError("acceptance.gate.control_modes must contain at least one mode.")
        if not self.rules:
            raise StudyConfigError("acceptance.gate.rules must declare every mandatory and descriptive pair.")
        if not any(rule.role == "mandatory_power" for rule in self.rules):
            raise StudyConfigError("acceptance.gate.rules must declare at least one 'mandatory_power' rule.")
        seen: set[tuple[str, str]] = set()
        for rule in self.rules:
            key = (rule.trajectory_mode, rule.statistic)
            if key in seen:
                raise StudyConfigError(f"acceptance.gate.rules has a duplicate entry for {key}.")
            seen.add(key)

    def power_floor(self, rule: GateRule) -> float:
        """Minimum power at the top effect for ``rule``."""

        return float(self.min_power_at_top if rule.min_power_at_top is None else rule.min_power_at_top)


@dataclass(frozen=True)
class AcceptanceTargets:
    """Collection of pre-specified acceptance targets."""

    type_i: tuple[TypeIControlTarget, ...] = ()
    power: tuple[PowerMonotonicityTarget, ...] = ()
    specificity: tuple[SpecificityTarget, ...] = ()
    gate: Phase4GateConfig = field(default_factory=Phase4GateConfig)


@dataclass(frozen=True)
class StudyConfig:
    """Declarative study definition."""

    generator: SemiSyntheticTrajectoryParams
    evaluation: SimulationEvaluationParams
    trajectory_modes: tuple[str, ...]
    effect_sizes: tuple[float, ...]
    axes: Mapping[str, tuple[Any, ...]] = field(default_factory=dict)
    n_replicates: int = 1
    base_seed: int = 0
    alpha: float = 0.05
    acceptance: AcceptanceTargets = field(default_factory=AcceptanceTargets)
    attribution: AttributionSelector = field(default_factory=AttributionSelector)
    matched_seeds: MatchedSeedPolicy = field(default_factory=MatchedSeedPolicy)
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
        if self.attribution.enabled:
            if self.evaluation.integration_method != "pls":
                raise StudyConfigError(
                    "attribution diagnostics require evaluation.integration_method='pls'; "
                    f"got {self.evaluation.integration_method!r}."
                )
            unselectable = sorted(set(self.attribution.trajectory_modes) - set(self.trajectory_modes))
            if unselectable:
                raise StudyConfigError(
                    f"attribution.trajectory_modes select mode(s) absent from trajectory_modes: {unselectable}."
                )
            if self.attribution.effect_sizes is not None:
                declared = [float(value) for value in self.effect_sizes]
                missing = [
                    float(value)
                    for value in self.attribution.effect_sizes
                    if not any(abs(float(value) - candidate) < 1e-12 for candidate in declared)
                ]
                if missing:
                    raise StudyConfigError(
                        f"attribution.effect_sizes select effect(s) absent from effect_sizes: {missing}."
                    )


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
    required = {"generator", "evaluation", "trajectory_modes", "effect_sizes"}
    missing = sorted(required - set(data))
    if missing:
        raise StudyConfigError(f"Study configuration is missing required field(s): {missing}.")

    generator = _build_generator(data["generator"])
    evaluation = _build_evaluation(data.get("evaluation") or {})
    trajectory_modes = tuple(str(mode) for mode in data["trajectory_modes"])
    effect_sizes = tuple(float(value) for value in data["effect_sizes"])
    axes = _build_axes(data.get("axes") or {})
    n_replicates = int(data.get("n_replicates", 1))
    base_seed = int(data.get("base_seed", 0))
    alpha = float(data.get("alpha", 0.05))
    acceptance = _build_acceptance(data.get("acceptance") or {})
    attribution = _build_attribution(data.get("attribution") or {})
    matched_seeds = _build_matched_seeds(data.get("matched_seeds") or {})
    metadata = dict(data.get("metadata") or {})
    return StudyConfig(
        generator=generator,
        evaluation=evaluation,
        trajectory_modes=trajectory_modes,
        effect_sizes=effect_sizes,
        axes=axes,
        n_replicates=n_replicates,
        base_seed=base_seed,
        alpha=alpha,
        acceptance=acceptance,
        attribution=attribution,
        matched_seeds=matched_seeds,
        metadata=metadata,
    )


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
        if f.name == "stage_sample_prop" and value is not None:
            value = tuple(float(v) for v in value)
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
        value = raw[f.name]
        if f.name == "attribution" and value is not None and not isinstance(value, AttributionDiagnosticSettings):
            value = _build_attribution_settings(value)
        kwargs[f.name] = value
    return SimulationEvaluationParams(**kwargs)


def _build_attribution_settings(raw: Mapping[str, Any]) -> AttributionDiagnosticSettings:
    field_names = {f.name for f in dataclasses.fields(AttributionDiagnosticSettings)}
    unknown = sorted(set(raw) - field_names)
    if unknown:
        raise StudyConfigError(f"evaluation.attribution has unknown field(s): {unknown}.")
    return AttributionDiagnosticSettings(**{key: raw[key] for key in raw})


def _build_axes(raw: Mapping[str, Any]) -> Mapping[str, tuple[Any, ...]]:
    axes: dict[str, tuple[Any, ...]] = {}
    for axis, values in raw.items():
        _validate_axis_namespace(axis)
        if not isinstance(values, Sequence) or isinstance(values, str | bytes):
            raise StudyConfigError(f"axis {axis!r} values must be a sequence.")
        axes[axis] = tuple(values)
    return axes


def _build_acceptance(raw: Mapping[str, Any]) -> AcceptanceTargets:
    unknown = sorted(set(raw) - {"type_i", "power", "specificity", "gate"})
    if unknown:
        raise StudyConfigError(f"acceptance has unknown block(s): {unknown}.")
    type_i = tuple(_build_type_i_target(entry) for entry in raw.get("type_i", []) or [])
    power = tuple(_build_power_target(entry) for entry in raw.get("power", []) or [])
    specificity = tuple(_build_specificity_target(entry) for entry in raw.get("specificity", []) or [])
    gate = _build_gate(raw.get("gate") or {})
    return AcceptanceTargets(type_i=type_i, power=power, specificity=specificity, gate=gate)


def _build_gate(raw: Mapping[str, Any]) -> Phase4GateConfig:
    if not raw:
        return Phase4GateConfig()
    known = {
        "enabled",
        "alpha",
        "control_se_tolerance",
        "monotonicity_se_tolerance",
        "min_power_at_top",
        "confirmation_se_threshold",
        "max_marginal_exceedances",
        "control_modes",
        "rules",
        "require_complete_records",
    }
    unknown = sorted(set(raw) - known)
    if unknown:
        raise StudyConfigError(f"acceptance.gate has unknown field(s): {unknown}.")
    enabled = bool(raw.get("enabled", True))
    if enabled:
        required = {
            "alpha",
            "control_se_tolerance",
            "monotonicity_se_tolerance",
            "min_power_at_top",
            "confirmation_se_threshold",
            "rules",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise StudyConfigError(f"acceptance.gate is missing required parameter(s): {missing}.")
    rules = tuple(_build_gate_rule(entry) for entry in raw.get("rules", []) or [])
    control_modes = (
        tuple(str(mode) for mode in raw["control_modes"])
        if "control_modes" in raw
        else ("none", "translation")
    )
    return Phase4GateConfig(
        enabled=enabled,
        alpha=float(raw.get("alpha", 0.05)),
        control_se_tolerance=float(raw.get("control_se_tolerance", 2.0)),
        monotonicity_se_tolerance=float(raw.get("monotonicity_se_tolerance", 2.0)),
        min_power_at_top=float(raw.get("min_power_at_top", 0.80)),
        confirmation_se_threshold=float(raw.get("confirmation_se_threshold", 1.0)),
        max_marginal_exceedances=int(raw.get("max_marginal_exceedances", 1)),
        control_modes=control_modes,
        rules=rules,
        require_complete_records=bool(raw.get("require_complete_records", True)),
    )


def _build_gate_rule(raw: Mapping[str, Any]) -> GateRule:
    required = {"trajectory_mode", "statistic", "role"}
    missing = sorted(required - set(raw))
    if missing:
        raise StudyConfigError(f"acceptance.gate.rules entry missing field(s): {missing}.")
    unknown = sorted(set(raw) - (required | {"min_power_at_top"}))
    if unknown:
        raise StudyConfigError(f"acceptance.gate.rules entry has unknown field(s): {unknown}.")
    floor = raw.get("min_power_at_top")
    return GateRule(
        trajectory_mode=str(raw["trajectory_mode"]),
        statistic=str(raw["statistic"]),
        role=str(raw["role"]),
        min_power_at_top=None if floor is None else float(floor),
    )


def _build_attribution(raw: Mapping[str, Any]) -> AttributionSelector:
    if not raw:
        return AttributionSelector()
    field_names = {f.name for f in dataclasses.fields(AttributionSelector)}
    unknown = sorted(set(raw) - field_names)
    if unknown:
        raise StudyConfigError(f"attribution has unknown field(s): {unknown}.")
    effect_sizes = raw.get("effect_sizes")
    return AttributionSelector(
        enabled=bool(raw.get("enabled", True)),
        trajectory_modes=tuple(str(mode) for mode in raw.get("trajectory_modes", ("orientation",))),
        phases=tuple(str(phase) for phase in raw.get("phases", ("power_primary",))),
        nonzero_effects_only=bool(raw.get("nonzero_effects_only", True)),
        effect_sizes=None if effect_sizes is None else tuple(float(value) for value in effect_sizes),
        bootstrap_replicates=int(raw.get("bootstrap_replicates", 100)),
        bootstrap_seed=None if raw.get("bootstrap_seed", 0) is None else int(raw.get("bootstrap_seed", 0)),
        top_k=int(raw.get("top_k", 20)),
        zero_tolerance=float(raw.get("zero_tolerance", 1e-12)),
    )


def _build_matched_seeds(raw: Mapping[str, Any]) -> MatchedSeedPolicy:
    if not raw:
        return MatchedSeedPolicy()
    field_names = {f.name for f in dataclasses.fields(MatchedSeedPolicy)}
    unknown = sorted(set(raw) - field_names)
    if unknown:
        raise StudyConfigError(f"matched_seeds has unknown field(s): {unknown}.")
    return MatchedSeedPolicy(
        enabled=bool(raw.get("enabled", True)),
        version=int(raw.get("version", 1)),
        primary_family=str(raw.get("primary_family", "primary")),
        shared_zero_effect_anchor=bool(raw.get("shared_zero_effect_anchor", True)),
    )


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
            f"axis {axis!r} must use a namespace prefix: 'generator.' or 'evaluation.'."
        )
    namespace, _, field_name = axis.partition(".")
    if namespace not in _AXIS_NAMESPACES:
        raise StudyConfigError(f"axis {axis!r} has unsupported namespace {namespace!r}.")
    if not field_name:
        raise StudyConfigError(f"axis {axis!r} is missing a field name.")


def _config_to_dict(config: StudyConfig) -> dict[str, Any]:
    return {
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
            "gate": _dataclass_dict(config.acceptance.gate),
        },
        "attribution": _dataclass_dict(config.attribution),
        "matched_seeds": _dataclass_dict(config.matched_seeds),
        "metadata": dict(config.metadata),
    }


def _dataclass_dict(value: Any) -> dict[str, Any]:
    return {f.name: _to_jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}


def _to_jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _dataclass_dict(value)
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, tuple | list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


__all__ = [
    "AcceptanceTargets",
    "AttributionSelector",
    "GateRule",
    "MatchedSeedPolicy",
    "Phase4GateConfig",
    "PowerMonotonicityTarget",
    "SpecificityTarget",
    "StudyConfig",
    "StudyConfigError",
    "TypeIControlTarget",
    "dump_study_config",
    "load_study_config",
]
