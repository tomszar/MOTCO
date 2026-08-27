"""Feature attribution for two-group trajectories in a shared PLS space.

The functions in this module deliberately condition on a fitted PLS estimator.
They explain an already detected orientation difference; they do not fit a
model, calculate a significance test, or replace observed features with their
low-rank reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np
import pandas as pd


class AttributionError(ValueError):
    """Raised when attribution inputs cannot satisfy the analysis contract."""


@dataclass(frozen=True)
class AttributionConfig:
    """Effective settings and coordinate-system metadata for an analysis."""

    group_column: str
    stage_column: str
    feature_names: tuple[str, ...]
    groups: tuple[str, str]
    stages: tuple[str, ...]
    mean_source: str
    input_units: str
    original_units_available: bool
    zero_tolerance: float
    bootstrap_replicates: int
    bootstrap_seed: int | None
    top_k: int
    model_components: int
    label_namespace: str | None
    label_source: str | None


@dataclass(frozen=True)
class ComponentAttribution:
    """Raw and directional values for one transition/component pair."""

    component: str
    group_transitions: tuple[np.ndarray, np.ndarray]
    path_lengths: tuple[float, float]
    unit_directions: tuple[np.ndarray | None, np.ndarray | None]
    directional_contrast: np.ndarray | None
    contrast_available: bool


@dataclass(frozen=True)
class TransitionAttribution:
    """Observed, PLS-captured, and residual attribution for one transition."""

    transition_id: str
    from_stage: str
    to_stage: str
    observed: ComponentAttribution
    pls_captured: ComponentAttribution
    residual: ComponentAttribution


@dataclass(frozen=True)
class FeatureEffect:
    """One signed feature contribution in a transition/component."""

    transition_id: str
    component: str
    feature: str
    effect_standardized: float
    effect_original: float | None


@dataclass(frozen=True)
class AggregateEffect:
    """One caller-label aggregate in a transition/component."""

    transition_id: str
    component: str
    label: str
    signed_effect: float
    absolute_effect: float
    feature_count: int
    label_namespace: str


@dataclass(frozen=True)
class BootstrapSummary:
    """Per-feature stability summary for one transition/component."""

    transition_id: str
    component: str
    requested_replicates: int
    valid_replicates: int
    top_k: int
    seed: int | None
    sign_stability: np.ndarray
    rank_stability: np.ndarray
    top_k_selection_frequency: np.ndarray
    nonzero_sign_replicates: np.ndarray


@dataclass(frozen=True)
class InterpretationMetadata:
    """Interpretation boundary recorded with every result."""

    statement: str
    causal_boundary: str
    significance_boundary: str
    reconstruction_boundary: str
    bootstrap_boundary: str
    label_source: str | None


@dataclass(frozen=True)
class OrientationAttributionResult:
    """Structured output of :func:`analyze_orientation_attribution`."""

    config: AttributionConfig
    means: pd.DataFrame
    reconstructed_means: pd.DataFrame
    transitions: tuple[TransitionAttribution, ...]
    feature_effects: pd.DataFrame
    transition_summaries: pd.DataFrame
    transition_vectors: pd.DataFrame
    aggregate_effects: pd.DataFrame
    bootstrap_summaries: pd.DataFrame
    interpretation: InterpretationMetadata
    feature_records: tuple[FeatureEffect, ...]
    aggregate_records: tuple[AggregateEffect, ...]
    bootstrap_records: tuple[BootstrapSummary, ...]

    @property
    def groups(self) -> tuple[str, str]:
        """Ordered group labels."""

        return self.config.groups

    @property
    def stages(self) -> tuple[str, ...]:
        """Ordered stage labels."""

        return self.config.stages

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Canonical feature order used by the model and result tables."""

        return self.config.feature_names

    def frames(self) -> dict[str, pd.DataFrame]:
        """Return findings-ready tabular views without prose parsing."""

        return attribution_frames(self)


AttributionResult = OrientationAttributionResult


def analyze_orientation_attribution(
    X: pd.DataFrame,
    metadata: pd.DataFrame,
    model: Any,
    *,
    group_col: str = "group",
    stage_col: str = "stage",
    feature_cols: Sequence[str] | None = None,
    groups: Sequence[str] | None = None,
    stages: Sequence[str] | None = None,
    mean_table: pd.DataFrame | None = None,
    precomputed_means: pd.DataFrame | None = None,
    original_scales: Sequence[float] | pd.Series | Mapping[str, float] | None = None,
    feature_scales: Sequence[float] | pd.Series | Mapping[str, float] | None = None,
    feature_groups: Mapping[str, str] | pd.Series | pd.DataFrame | None = None,
    group_labels: Mapping[str, str] | pd.Series | pd.DataFrame | None = None,
    label_namespace: str = "module",
    bootstrap_replicates: int = 0,
    bootstrap_seed: int | None = 0,
    top_k: int = 10,
    zero_tolerance: float = 1e-12,
    sample_id_col: str | None = None,
) -> OrientationAttributionResult:
    """Attribute adjacent two-group trajectory orientation differences.

    Parameters
    ----------
    X:
        Aligned feature matrix in the exact coordinate system consumed by
        ``model``. Observed values are never overwritten by reconstructions.
    metadata:
        Row-aligned metadata containing ``group_col`` and ``stage_col``.
    model:
        One already fitted pooled PLS estimator exposing ``transform`` and
        ``inverse_transform``. It is reused for all groups and replicates.
    mean_table, precomputed_means:
        Optional complete group-by-stage table. It may have a two-level
        MultiIndex or explicit group/stage columns. When supplied, bootstrap
        resampling is unavailable because row-level sampling semantics are not
        defined for adjusted means.
    original_scales, feature_scales:
        Optional positive per-feature scale in canonical feature order. The
        aliases are accepted for readability at call sites; supplying both is
        an error.
    feature_groups, group_labels:
        Optional one-to-one caller-supplied feature-to-label mapping. The
        aliases are accepted for readability at call sites.

    Returns
    -------
    OrientationAttributionResult
        Frozen typed records and DataFrame views for means, transitions,
        feature effects, aggregates, and bootstrap stability.
    """
    if mean_table is not None and precomputed_means is not None:
        raise AttributionError("Provide only one of mean_table and precomputed_means.")
    if original_scales is not None and feature_scales is not None:
        raise AttributionError("Provide only one of original_scales and feature_scales.")
    if feature_groups is not None and group_labels is not None:
        raise AttributionError("Provide only one of feature_groups and group_labels.")
    if not isinstance(label_namespace, str) or not label_namespace.strip():
        raise AttributionError("label_namespace must be a non-empty string.")
    if precomputed_means is not None:
        mean_table = precomputed_means
    if feature_scales is not None:
        original_scales = feature_scales
    if group_labels is not None:
        feature_groups = group_labels

    feature_frame, metadata_frame, feature_names = _validate_inputs(
        X,
        metadata,
        model,
        group_col,
        stage_col,
        feature_cols,
        sample_id_col,
    )
    ordered_groups = cast(
        tuple[str, str], _ordered_levels(metadata_frame[group_col], groups, 2, "groups")
    )
    ordered_stages = _ordered_levels(metadata_frame[stage_col], stages, 2, "stages")
    _validate_cells(metadata_frame, group_col, stage_col, ordered_groups, ordered_stages)
    scale_values = _validate_scales(original_scales, feature_names)
    label_values = _validate_feature_groups(feature_groups, feature_names)

    if mean_table is None:
        means_array = _arithmetic_means(
            feature_frame,
            metadata_frame,
            group_col,
            stage_col,
            ordered_groups,
            ordered_stages,
        )
        mean_source = "arithmetic"
    else:
        means_array = _precomputed_mean_array(
            mean_table,
            feature_names,
            ordered_groups,
            ordered_stages,
            group_col,
            stage_col,
        )
        mean_source = "precomputed"
        if bootstrap_replicates:
            raise AttributionError(
                "bootstrap_replicates must be zero when mean_table contains precomputed means; "
                "row-level bootstrap semantics are unavailable."
            )

    _validate_bootstrap_settings(bootstrap_replicates, top_k, zero_tolerance)
    reconstructed_array = _reconstruct_means(means_array, model, feature_names)
    transitions = _build_transitions(
        means_array,
        reconstructed_array,
        ordered_groups,
        ordered_stages,
        zero_tolerance,
    )
    model_components = _model_components(model)
    config = AttributionConfig(
        group_column=group_col,
        stage_column=stage_col,
        feature_names=feature_names,
        groups=ordered_groups,
        stages=ordered_stages,
        mean_source=mean_source,
        input_units="standardized_input",
        original_units_available=scale_values is not None,
        zero_tolerance=float(zero_tolerance),
        bootstrap_replicates=int(bootstrap_replicates),
        bootstrap_seed=bootstrap_seed,
        top_k=int(top_k),
        model_components=model_components,
        label_namespace=label_namespace if label_values is not None else None,
        label_source="caller-supplied" if label_values is not None else None,
    )

    bootstrap_records = _bootstrap(
        feature_frame,
        metadata_frame,
        group_col,
        stage_col,
        ordered_groups,
        ordered_stages,
        model,
        feature_names,
        scale_values,
        bootstrap_replicates,
        bootstrap_seed,
        top_k,
        zero_tolerance,
    )
    bootstrap_lookup = {
        (record.transition_id, record.component): record for record in bootstrap_records
    }
    feature_effects, feature_records = _feature_effect_table(
        transitions,
        feature_names,
        scale_values,
        bootstrap_lookup,
        label_values,
        label_namespace,
        bootstrap_replicates,
    )
    aggregate_effects, aggregate_records = _aggregate_effect_table(
        feature_effects,
        feature_names,
        label_values,
        label_namespace,
        bootstrap_replicates,
    )
    transition_summaries, transition_vectors = _transition_tables(
        transitions,
        ordered_groups,
        feature_names,
    )
    bootstrap_table = _bootstrap_table(bootstrap_records, feature_names)
    means_table = _means_table(means_array, ordered_groups, ordered_stages, feature_names)
    reconstructed_table = _means_table(
        reconstructed_array, ordered_groups, ordered_stages, feature_names
    )
    interpretation = InterpretationMetadata(
        statement="Attribution describes associations implied by one fitted shared PLS representation.",
        causal_boundary="Feature, module, and pathway effects are not causal claims.",
        significance_boundary=(
            "No p-values or significance decisions are produced; external orientation inference remains the caller's "
            "responsibility."
        ),
        reconstruction_boundary=(
            "PLS-captured effects are a lossy inverse-transform view and are not a unique inverse of measured features."
        ),
        bootstrap_boundary=(
            "Bootstrap stability is conditional on the fitted preprocessing contract and frozen PLS model."
        ),
        label_source="caller-supplied" if label_values is not None else None,
    )
    return OrientationAttributionResult(
        config=config,
        means=means_table,
        reconstructed_means=reconstructed_table,
        transitions=tuple(transitions),
        feature_effects=feature_effects,
        transition_summaries=transition_summaries,
        transition_vectors=transition_vectors,
        aggregate_effects=aggregate_effects,
        bootstrap_summaries=bootstrap_table,
        interpretation=interpretation,
        feature_records=tuple(feature_records),
        aggregate_records=tuple(aggregate_records),
        bootstrap_records=tuple(bootstrap_records),
    )


run_orientation_attribution = analyze_orientation_attribution


def attribution_frames(result: OrientationAttributionResult) -> dict[str, pd.DataFrame]:
    """Return all findings-ready tables for ``result``."""

    configuration = pd.DataFrame(
        [
            ("group_column", result.config.group_column),
            ("stage_column", result.config.stage_column),
            ("feature_names", list(result.config.feature_names)),
            ("groups", list(result.config.groups)),
            ("stages", list(result.config.stages)),
            ("mean_source", result.config.mean_source),
            ("input_units", result.config.input_units),
            ("original_units_available", result.config.original_units_available),
            ("zero_tolerance", result.config.zero_tolerance),
            ("bootstrap_replicates", result.config.bootstrap_replicates),
            ("bootstrap_seed", result.config.bootstrap_seed),
            ("top_k", result.config.top_k),
            ("model_components", result.config.model_components),
            ("label_namespace", result.config.label_namespace),
            ("label_source", result.config.label_source),
        ],
        columns=["setting", "value"],
    )
    interpretation = pd.DataFrame(
        [
            ("statement", result.interpretation.statement),
            ("causal_boundary", result.interpretation.causal_boundary),
            ("significance_boundary", result.interpretation.significance_boundary),
            ("reconstruction_boundary", result.interpretation.reconstruction_boundary),
            ("bootstrap_boundary", result.interpretation.bootstrap_boundary),
            ("label_source", result.interpretation.label_source),
        ],
        columns=["field", "value"],
    )
    return {
        "means": result.means.copy(),
        "reconstructed_means": result.reconstructed_means.copy(),
        "feature_effects": result.feature_effects.copy(),
        "transition_summaries": result.transition_summaries.copy(),
        "transition_vectors": result.transition_vectors.copy(),
        "aggregate_effects": result.aggregate_effects.copy(),
        "bootstrap_summaries": result.bootstrap_summaries.copy(),
        "configuration": configuration,
        "interpretation": interpretation,
    }


def write_attribution_outputs(
    result: OrientationAttributionResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write attribution tables as CSV files and return their paths."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, frame in attribution_frames(result).items():
        path = directory / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
    return paths


def _validate_inputs(
    X: pd.DataFrame,
    metadata: pd.DataFrame,
    model: Any,
    group_col: str,
    stage_col: str,
    feature_cols: Sequence[str] | None,
    sample_id_col: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    if not isinstance(X, pd.DataFrame):
        raise AttributionError("X must be a pandas DataFrame with canonical feature columns.")
    if not isinstance(metadata, pd.DataFrame):
        raise AttributionError("metadata must be a pandas DataFrame aligned to X rows.")
    if not X.index.equals(metadata.index):
        raise AttributionError("X and metadata rows or sample identifiers are not aligned.")
    if X.index.has_duplicates or metadata.index.has_duplicates:
        raise AttributionError("X and metadata sample identifiers must be unique.")
    if group_col not in metadata.columns or stage_col not in metadata.columns:
        raise AttributionError(
            f"metadata must contain group column '{group_col}' and stage column '{stage_col}'."
        )
    if sample_id_col is not None:
        if sample_id_col not in metadata.columns:
            raise AttributionError(f"sample_id_col='{sample_id_col}' is not present in metadata.")
        sample_ids = metadata[sample_id_col]
        if sample_ids.isna().any() or sample_ids.duplicated().any():
            raise AttributionError("sample identifiers must be non-missing and unique.")
        if not pd.Index(sample_ids.tolist()).equals(X.index):
            raise AttributionError("sample identifiers do not match the feature matrix index.")

    names = tuple(map(str, feature_cols if feature_cols is not None else X.columns.tolist()))
    if len(names) == 0:
        raise AttributionError("At least one feature column is required.")
    if len(set(names)) != len(names):
        raise AttributionError("feature names must be unique.")
    actual_names = tuple(map(str, X.columns.tolist()))
    if actual_names != names:
        missing = [name for name in names if name not in actual_names]
        if missing:
            raise AttributionError(f"feature_cols contains unknown feature(s): {missing}.")
        positions = [actual_names.index(name) for name in names]
        X = X.iloc[:, positions]
    try:
        feature_frame = X.copy()
        feature_frame.columns = names
        feature_frame = feature_frame.astype(float)
    except (TypeError, ValueError) as exc:
        raise AttributionError("X feature columns must be numeric.") from exc
    if not np.isfinite(feature_frame.to_numpy()).all():
        raise AttributionError("X contains NaN or Inf values.")

    n_features = len(names)
    model_features = getattr(model, "n_features_in_", None)
    if model_features is not None and int(model_features) != n_features:
        raise AttributionError(
            f"fitted model expects {int(model_features)} features but X has {n_features}."
        )
    model_names = getattr(model, "feature_names_in_", None)
    if model_names is not None and tuple(map(str, model_names)) != names:
        raise AttributionError("X feature order does not match the fitted model feature order.")
    if not callable(getattr(model, "transform", None)) or not callable(
        getattr(model, "inverse_transform", None)
    ):
        raise AttributionError("fitted model must expose transform and inverse_transform methods.")
    return feature_frame, metadata.copy(), names


def _ordered_levels(
    values: pd.Series,
    explicit: Sequence[str] | None,
    minimum: int,
    name: str,
) -> tuple[str, ...]:
    if values.isna().any():
        raise AttributionError(f"{name} contain missing values.")
    observed = {str(value) for value in pd.unique(values)}
    if explicit is None:
        ordered = tuple(sorted(observed))
    else:
        ordered = tuple(map(str, explicit))
        if len(set(ordered)) != len(ordered):
            raise AttributionError(f"explicit {name} must not contain duplicates.")
        if set(ordered) != observed:
            raise AttributionError(
                f"explicit {name} {ordered} do not match observed {name} {tuple(sorted(observed))}."
            )
    if len(ordered) < minimum:
        raise AttributionError(f"expected at least {minimum} {name}; found {len(ordered)}: {ordered}.")
    if name == "groups" and len(ordered) != 2:
        raise AttributionError(f"expected exactly 2 groups; found {len(ordered)}: {ordered}.")
    return ordered


def _validate_cells(
    metadata: pd.DataFrame,
    group_col: str,
    stage_col: str,
    groups: tuple[str, str],
    stages: tuple[str, ...],
) -> None:
    group_values = metadata[group_col].astype(str)
    stage_values = metadata[stage_col].astype(str)
    missing = [
        (group, stage)
        for group in groups
        for stage in stages
        if not ((group_values == group) & (stage_values == stage)).any()
    ]
    if missing:
        raise AttributionError(f"missing complete group-by-stage cells: {missing}.")


def _validate_scales(
    scales: Sequence[float] | pd.Series | Mapping[str, float] | None,
    feature_names: tuple[str, ...],
) -> np.ndarray | None:
    if scales is None:
        return None
    if isinstance(scales, pd.Series):
        if not scales.index.is_unique or tuple(map(str, scales.index)) != feature_names:
            raise AttributionError("feature scales must have unique index in the input feature order.")
        values = scales.to_numpy(dtype=float)
    elif isinstance(scales, Mapping):
        keys = tuple(map(str, scales.keys()))
        if keys != feature_names:
            raise AttributionError("feature scale mapping keys must match the input feature order.")
        values = np.asarray([scales[name] for name in feature_names], dtype=float)
    else:
        values = np.asarray(scales, dtype=float)
        if values.ndim != 1 or len(values) != len(feature_names):
            raise AttributionError("feature scales must contain exactly one value per input feature.")
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise AttributionError("feature scales must be finite and strictly positive.")
    return values


def _validate_feature_groups(
    labels: Mapping[str, str] | pd.Series | pd.DataFrame | None,
    feature_names: tuple[str, ...],
) -> dict[str, str] | None:
    if labels is None:
        return None
    pairs: list[tuple[Any, Any]]
    if isinstance(labels, pd.Series):
        pairs = list(zip(labels.index.tolist(), labels.tolist()))
    elif isinstance(labels, Mapping):
        pairs = list(labels.items())
    elif isinstance(labels, pd.DataFrame):
        if labels.shape[1] != 2:
            raise AttributionError("feature-group DataFrame must contain exactly feature and label columns.")
        pairs = list(zip(labels.iloc[:, 0].tolist(), labels.iloc[:, 1].tolist()))
    else:
        raise AttributionError("feature groups must be a mapping, Series, or two-column DataFrame.")
    result: dict[str, str] = {}
    for feature, label in pairs:
        feature_name = str(feature)
        if feature_name not in feature_names:
            raise AttributionError(f"feature-group mapping contains unknown feature '{feature_name}'.")
        if pd.isna(label) or str(label).strip() == "":
            raise AttributionError(f"feature-group mapping has an empty label for '{feature_name}'.")
        label_name = str(label)
        if feature_name in result and result[feature_name] != label_name:
            raise AttributionError(f"feature '{feature_name}' has conflicting group labels.")
        result[feature_name] = label_name
    missing = [feature for feature in feature_names if feature not in result]
    if missing:
        raise AttributionError(f"feature-group mapping omits input feature(s): {missing}.")
    return result


def _arithmetic_means(
    X: pd.DataFrame,
    metadata: pd.DataFrame,
    group_col: str,
    stage_col: str,
    groups: tuple[str, str],
    stages: tuple[str, ...],
) -> np.ndarray:
    group_values = metadata[group_col].astype(str)
    stage_values = metadata[stage_col].astype(str)
    values = X.to_numpy(dtype=float)
    result = np.empty((len(groups), len(stages), X.shape[1]), dtype=float)
    for gi, group in enumerate(groups):
        for si, stage in enumerate(stages):
            mask = ((group_values == group) & (stage_values == stage)).to_numpy()
            result[gi, si] = values[mask].mean(axis=0)
    return result


def _precomputed_mean_array(
    table: pd.DataFrame,
    feature_names: tuple[str, ...],
    groups: tuple[str, str],
    stages: tuple[str, ...],
    group_col: str,
    stage_col: str,
) -> np.ndarray:
    if not isinstance(table, pd.DataFrame):
        raise AttributionError("mean_table must be a pandas DataFrame.")
    frame = table.copy()
    if group_col in frame.columns and stage_col in frame.columns:
        group_values = frame[group_col].astype(str)
        stage_values = frame[stage_col].astype(str)
        table_features = [column for column in frame.columns if column not in {group_col, stage_col}]
        if tuple(map(str, table_features)) != feature_names:
            raise AttributionError("mean_table feature columns must match the input feature order exactly.")
        values = frame.loc[:, table_features].to_numpy(dtype=float)
    elif isinstance(frame.index, pd.MultiIndex) and frame.index.nlevels == 2:
        group_values = pd.Series(frame.index.get_level_values(0).astype(str), index=frame.index)
        stage_values = pd.Series(frame.index.get_level_values(1).astype(str), index=frame.index)
        if tuple(map(str, frame.columns)) != feature_names:
            raise AttributionError("mean_table feature columns must match the input feature order exactly.")
        values = frame.loc[:, list(feature_names)].to_numpy(dtype=float)
    else:
        raise AttributionError(
            f"mean_table must have '{group_col}'/'{stage_col}' columns or a two-level MultiIndex."
        )
    if not np.isfinite(values).all():
        raise AttributionError("mean_table contains NaN or Inf values.")
    result = np.empty((len(groups), len(stages), len(feature_names)), dtype=float)
    for gi, group in enumerate(groups):
        for si, stage in enumerate(stages):
            mask = ((group_values == group) & (stage_values == stage)).to_numpy()
            if int(mask.sum()) != 1:
                raise AttributionError(
                    f"mean_table must contain exactly one row for group-stage cell ({group}, {stage})."
                )
            result[gi, si] = values[mask][0]
    if set(zip(group_values, stage_values)) != {(g, s) for g in groups for s in stages}:
        raise AttributionError("mean_table contains unknown or incomplete group-by-stage cells.")
    return result


def _reconstruct_means(means: np.ndarray, model: Any, feature_names: tuple[str, ...]) -> np.ndarray:
    flat = means.reshape((-1, len(feature_names)))
    try:
        scores = np.asarray(model.transform(flat), dtype=float)
        reconstructed = np.asarray(model.inverse_transform(scores), dtype=float)
    except Exception as exc:  # sklearn estimators expose varied exception types
        raise AttributionError(
            "fitted model could not transform and inverse-transform the aligned feature means."
        ) from exc
    if reconstructed.shape != flat.shape or not np.isfinite(reconstructed).all():
        raise AttributionError("fitted model returned invalid inverse-transformed mean dimensions or values.")
    return reconstructed.reshape(means.shape)


def _model_components(model: Any) -> int:
    value = getattr(model, "n_components_", getattr(model, "n_components", None))
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _component(
    name: str,
    transitions: tuple[np.ndarray, np.ndarray],
    tolerance: float,
    contrast: np.ndarray | None = None,
    derive_contrast: bool = True,
) -> ComponentAttribution:
    lengths: tuple[float, float] = (
        float(np.linalg.norm(transitions[0])),
        float(np.linalg.norm(transitions[1])),
    )
    directions: tuple[np.ndarray | None, np.ndarray | None] = (
        transitions[0] / lengths[0] if lengths[0] > tolerance else None,
        transitions[1] / lengths[1] if lengths[1] > tolerance else None,
    )
    directions_available = directions[0] is not None and directions[1] is not None
    if contrast is None and derive_contrast and directions_available:
        assert directions[0] is not None and directions[1] is not None
        contrast = directions[1] - directions[0]
    available = contrast is not None
    return ComponentAttribution(
        component=name,
        group_transitions=transitions,
        path_lengths=lengths,
        unit_directions=directions,
        directional_contrast=contrast,
        contrast_available=available,
    )


def _build_transitions(
    observed: np.ndarray,
    reconstructed: np.ndarray,
    groups: tuple[str, str],
    stages: tuple[str, ...],
    tolerance: float,
) -> list[TransitionAttribution]:
    del groups  # The ordered group contract is encoded by tuple position.
    records: list[TransitionAttribution] = []
    for stage_index in range(len(stages) - 1):
        from_stage, to_stage = stages[stage_index : stage_index + 2]
        transition_id = f"{from_stage}->{to_stage}"
        observed_d = tuple(observed[group, stage_index + 1] - observed[group, stage_index] for group in range(2))
        captured_d = tuple(
            reconstructed[group, stage_index + 1] - reconstructed[group, stage_index] for group in range(2)
        )
        residual_d = tuple(observed_d[index] - captured_d[index] for index in range(2))
        observed_component = _component("observed", observed_d, tolerance)
        captured_component = _component("pls_captured", captured_d, tolerance)
        residual_contrast = None
        if observed_component.directional_contrast is not None and captured_component.directional_contrast is not None:
            residual_contrast = observed_component.directional_contrast - captured_component.directional_contrast
        residual_component = _component(
            "residual", residual_d, tolerance, residual_contrast, derive_contrast=False
        )
        records.append(
            TransitionAttribution(
                transition_id=transition_id,
                from_stage=from_stage,
                to_stage=to_stage,
                observed=observed_component,
                pls_captured=captured_component,
                residual=residual_component,
            )
        )
    return records


def _bootstrap(
    X: pd.DataFrame,
    metadata: pd.DataFrame,
    group_col: str,
    stage_col: str,
    groups: tuple[str, str],
    stages: tuple[str, ...],
    model: Any,
    feature_names: tuple[str, ...],
    scales: np.ndarray | None,
    replicates: int,
    seed: int | None,
    top_k: int,
    tolerance: float,
) -> tuple[BootstrapSummary, ...]:
    del scales  # Bootstrap stability is summarized in the fixed input units.
    if replicates == 0:
        return tuple(
            BootstrapSummary(
                transition_id=f"{stages[index]}->{stages[index + 1]}",
                component=component,
                requested_replicates=0,
                valid_replicates=0,
                top_k=top_k,
                seed=seed,
                sign_stability=np.full(len(feature_names), np.nan),
                rank_stability=np.full(len(feature_names), np.nan),
                top_k_selection_frequency=np.full(len(feature_names), np.nan),
                nonzero_sign_replicates=np.zeros(len(feature_names), dtype=int),
            )
            for index in range(len(stages) - 1)
            for component in ("observed", "pls_captured", "residual")
        )
    group_values = metadata[group_col].astype(str).to_numpy()
    stage_values = metadata[stage_col].astype(str).to_numpy()
    strata = [
        np.flatnonzero((group_values == group) & (stage_values == stage))
        for group in groups
        for stage in stages
    ]
    rng = np.random.default_rng(seed)
    per_component: dict[tuple[str, str], list[np.ndarray]] = {}
    for _ in range(replicates):
        sampled = np.empty((len(groups), len(stages), X.shape[1]), dtype=float)
        for cell_index, source_indices in enumerate(strata):
            selected = rng.choice(source_indices, size=len(source_indices), replace=True)
            sampled[cell_index // len(stages), cell_index % len(stages)] = X.iloc[selected].mean(axis=0).to_numpy()
        reconstructed = _reconstruct_means(sampled, model, feature_names)
        transitions = _build_transitions(sampled, reconstructed, groups, stages, tolerance)
        for record in transitions:
            for component_name in ("observed", "pls_captured", "residual"):
                component = getattr(record, component_name)
                key = (record.transition_id, component_name)
                per_component.setdefault(key, []).append(
                    component.directional_contrast
                    if component.directional_contrast is not None
                    else np.full(len(feature_names), np.nan)
                )
    summaries: list[BootstrapSummary] = []
    for transition_index in range(len(stages) - 1):
        transition_id = f"{stages[transition_index]}->{stages[transition_index + 1]}"
        for component_name in ("observed", "pls_captured", "residual"):
            values = np.asarray(per_component[(transition_id, component_name)], dtype=float)
            valid_mask = np.isfinite(values).all(axis=1)
            valid_values = values[valid_mask]
            valid_count = len(valid_values)
            sign_stability = np.full(len(feature_names), np.nan)
            rank_stability = np.full(len(feature_names), np.nan)
            top_frequency = np.full(len(feature_names), np.nan)
            nonzero_counts = np.zeros(len(feature_names), dtype=int)
            selected_counts = np.zeros(len(feature_names), dtype=int)
            for vector in valid_values:
                nonzero = np.abs(vector) > tolerance
                nonzero_counts += nonzero
                if nonzero.any():
                    selected = np.argsort(-np.where(nonzero, np.abs(vector), -np.inf), kind="stable")[:top_k]
                    selected = selected[nonzero[selected]]
                    selected_counts[selected] += 1
            if valid_count:
                top_frequency = selected_counts / valid_count
                rank_stability = top_frequency.copy()
                for feature_index in range(len(feature_names)):
                    nonzero_vectors = valid_values[:, feature_index]
                    nonzero_vectors = nonzero_vectors[np.abs(nonzero_vectors) > tolerance]
                    nonzero_counts[feature_index] = len(nonzero_vectors)
                    if len(nonzero_vectors):
                        modal_sign = 1.0 if np.sum(nonzero_vectors > 0) >= np.sum(nonzero_vectors < 0) else -1.0
                        sign_stability[feature_index] = float(
                            np.sum(np.sign(nonzero_vectors) == modal_sign) / len(nonzero_vectors)
                        )
            summaries.append(
                BootstrapSummary(
                    transition_id=transition_id,
                    component=component_name,
                    requested_replicates=replicates,
                    valid_replicates=valid_count,
                    top_k=top_k,
                    seed=seed,
                    sign_stability=sign_stability,
                    rank_stability=rank_stability,
                    top_k_selection_frequency=top_frequency,
                    nonzero_sign_replicates=nonzero_counts,
                )
            )
    return tuple(summaries)


def _validate_bootstrap_settings(replicates: int, top_k: int, tolerance: float) -> None:
    if isinstance(replicates, bool) or replicates < 0:
        raise AttributionError("bootstrap_replicates must be a non-negative integer.")
    if int(replicates) != replicates:
        raise AttributionError("bootstrap_replicates must be a non-negative integer.")
    if isinstance(top_k, bool) or top_k < 1:
        raise AttributionError("top_k must be a positive integer.")
    if int(top_k) != top_k:
        raise AttributionError("top_k must be a positive integer.")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise AttributionError("zero_tolerance must be finite and non-negative.")


def _component_effect(component: ComponentAttribution, p: int) -> np.ndarray:
    if component.directional_contrast is None:
        return np.full(p, np.nan)
    return component.directional_contrast


def _feature_effect_table(
    transitions: Sequence[TransitionAttribution],
    feature_names: tuple[str, ...],
    scales: np.ndarray | None,
    bootstrap: Mapping[tuple[str, str], BootstrapSummary],
    labels: Mapping[str, str] | None,
    label_namespace: str,
    requested_replicates: int,
) -> tuple[pd.DataFrame, list[FeatureEffect]]:
    rows: list[dict[str, Any]] = []
    records: list[FeatureEffect] = []
    for transition in transitions:
        for component_name in ("observed", "pls_captured", "residual"):
            component = getattr(transition, component_name)
            effects = _component_effect(component, len(feature_names))
            summary = bootstrap[(transition.transition_id, component_name)]
            for index, feature in enumerate(feature_names):
                standardized = float(effects[index])
                original = (
                    None
                    if scales is None or not np.isfinite(standardized)
                    else float(standardized * scales[index])
                )
                records.append(
                    FeatureEffect(transition.transition_id, component_name, feature, standardized, original)
                )
                rows.append(
                    {
                        "transition_id": transition.transition_id,
                        "from_stage": transition.from_stage,
                        "to_stage": transition.to_stage,
                        "component": component_name,
                        "feature": feature,
                        "effect_standardized": standardized,
                        "effect_original": original,
                        "effect_unit": "standardized_input",
                        "feature_group": labels[feature] if labels is not None else None,
                        "label_namespace": label_namespace if labels is not None else None,
                        "label_source": "caller-supplied" if labels is not None else None,
                        "sign_stability": _array_value(summary.sign_stability, index),
                        "rank_stability": _array_value(summary.rank_stability, index),
                        "top_k_selection_frequency": _array_value(summary.top_k_selection_frequency, index),
                        "valid_bootstrap_replicates": summary.valid_replicates,
                        "nonzero_sign_replicates": int(summary.nonzero_sign_replicates[index]),
                        "requested_bootstrap_replicates": requested_replicates,
                    }
                )
    return pd.DataFrame(rows), records


def _aggregate_effect_table(
    feature_effects: pd.DataFrame,
    feature_names: tuple[str, ...],
    labels: Mapping[str, str] | None,
    label_namespace: str,
    requested_replicates: int,
) -> tuple[pd.DataFrame, list[AggregateEffect]]:
    del feature_names
    if labels is None:
        return pd.DataFrame(), []
    rows: list[dict[str, Any]] = []
    records: list[AggregateEffect] = []
    for (transition_id, component, label), group in feature_effects.groupby(
        ["transition_id", "component", "feature_group"], sort=False, dropna=False
    ):
        standardized = group["effect_standardized"].to_numpy(dtype=float)
        original = group["effect_original"].to_numpy(dtype=float)
        signed = float(np.nansum(standardized)) if np.isfinite(standardized).any() else np.nan
        absolute = float(np.nansum(np.abs(standardized))) if np.isfinite(standardized).any() else np.nan
        original_signed = float(np.nansum(original)) if np.isfinite(original).any() else np.nan
        original_absolute = float(np.nansum(np.abs(original))) if np.isfinite(original).any() else np.nan
        def mean_available(column: str) -> float:
            values = group[column].to_numpy(dtype=float)
            return float(np.nanmean(values)) if np.isfinite(values).any() else np.nan

        row = {
            "transition_id": transition_id,
            "component": component,
            "label": label,
            "label_namespace": label_namespace,
            "label_source": "caller-supplied",
            "signed_effect_standardized": signed,
            "absolute_effect_standardized": absolute,
            "signed_effect_original": original_signed,
            "absolute_effect_original": original_absolute,
            "feature_count": int(len(group)),
            "sign_stability": mean_available("sign_stability"),
            "rank_stability": mean_available("rank_stability"),
            "top_k_selection_frequency": mean_available("top_k_selection_frequency"),
            "valid_bootstrap_replicates": int(group["valid_bootstrap_replicates"].max()),
            "requested_bootstrap_replicates": requested_replicates,
        }
        rows.append(row)
        records.append(
            AggregateEffect(transition_id, component, str(label), signed, absolute, len(group), label_namespace)
        )
    return pd.DataFrame(rows), records


def _transition_tables(
    transitions: Sequence[TransitionAttribution],
    groups: tuple[str, str],
    feature_names: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    vector_rows: list[dict[str, Any]] = []
    for transition in transitions:
        for component_name in ("observed", "pls_captured", "residual"):
            component = getattr(transition, component_name)
            summary_rows.append(
                {
                    "transition_id": transition.transition_id,
                    "from_stage": transition.from_stage,
                    "to_stage": transition.to_stage,
                    "component": component_name,
                    "group_1": groups[0],
                    "group_2": groups[1],
                    "path_length_group_1": component.path_lengths[0],
                    "path_length_group_2": component.path_lengths[1],
                    "contrast_available": component.contrast_available,
                    "contrast_norm": (
                        float(np.linalg.norm(component.directional_contrast))
                        if component.directional_contrast is not None
                        else np.nan
                    ),
                }
            )
            for group_index, group in enumerate(groups):
                for feature_index, feature in enumerate(feature_names):
                    vector_rows.append(
                        {
                            "transition_id": transition.transition_id,
                            "from_stage": transition.from_stage,
                            "to_stage": transition.to_stage,
                            "component": component_name,
                            "group": group,
                            "feature": feature,
                            "raw_transition": float(component.group_transitions[group_index][feature_index]),
                        }
                    )
    return pd.DataFrame(summary_rows), pd.DataFrame(vector_rows)


def _bootstrap_table(records: Sequence[BootstrapSummary], feature_names: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        for index, feature in enumerate(feature_names):
            rows.append(
                {
                    "transition_id": record.transition_id,
                    "component": record.component,
                    "feature": feature,
                    "requested_replicates": record.requested_replicates,
                    "valid_replicates": record.valid_replicates,
                    "top_k": record.top_k,
                    "seed": record.seed,
                    "sign_stability": _array_value(record.sign_stability, index),
                    "rank_stability": _array_value(record.rank_stability, index),
                    "top_k_selection_frequency": _array_value(record.top_k_selection_frequency, index),
                    "nonzero_sign_replicates": int(record.nonzero_sign_replicates[index]),
                }
            )
    return pd.DataFrame(rows)


def _means_table(
    values: np.ndarray,
    groups: tuple[str, str],
    stages: tuple[str, ...],
    feature_names: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_index, group in enumerate(groups):
        for stage_index, stage in enumerate(stages):
            row: dict[str, Any] = {"group": group, "stage": stage}
            row.update(
                {
                    feature: float(values[group_index, stage_index, index])
                    for index, feature in enumerate(feature_names)
                }
            )
            rows.append(row)
    return pd.DataFrame(rows, columns=["group", "stage", *feature_names])


def _array_value(values: np.ndarray, index: int) -> float:
    value = values[index]
    return float(value) if np.isfinite(value) else np.nan


__all__ = [
    "AggregateEffect",
    "AttributionConfig",
    "AttributionError",
    "AttributionResult",
    "BootstrapSummary",
    "ComponentAttribution",
    "FeatureEffect",
    "InterpretationMetadata",
    "OrientationAttributionResult",
    "TransitionAttribution",
    "analyze_orientation_attribution",
    "attribution_frames",
    "run_orientation_attribution",
    "write_attribution_outputs",
]
