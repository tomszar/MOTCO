"""Semi-synthetic trajectory dataset generation from InterSIM outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from motco.simulations.intersim import InterSIMParams, InterSIMResult, run_intersim

OmicsLayer = Literal["methylation", "expression", "proteomics"]
TrajectoryMode = Literal["none", "translation", "magnitude", "orientation", "shape"]

_OMICS_LAYERS: tuple[OmicsLayer, ...] = ("methylation", "expression", "proteomics")


class SemiSyntheticTrajectoryError(ValueError):
    """Raised when semi-synthetic trajectory generation parameters are invalid."""


@dataclass(frozen=True)
class SemiSyntheticTrajectoryParams:
    """Parameters for semi-synthetic trajectory generation.

    ``group_ratio`` is the target proportion assigned to the first label in
    ``group_labels`` within every generated stage.
    """

    seed: int
    trajectory_mode: TrajectoryMode = "none"
    group_effect_size: float = 0.0
    group_ratio: float = 0.5
    group_labels: tuple[str, str] = ("A", "B")
    prop_affected_features: float | Mapping[OmicsLayer, float] = 0.1
    affected_features: Mapping[OmicsLayer, Sequence[str]] | None = None


@dataclass(frozen=True)
class SemiSyntheticTrajectoryDataset:
    """MOTCO-ready semi-synthetic trajectory dataset."""

    methylation: pd.DataFrame
    expression: pd.DataFrame
    proteomics: pd.DataFrame
    metadata: pd.DataFrame
    truth: dict[str, Any] = field(default_factory=dict)


def generate_semisynthetic_trajectory(
    intersim_result: InterSIMResult,
    params: SemiSyntheticTrajectoryParams,
) -> SemiSyntheticTrajectoryDataset:
    """Generate a semi-synthetic trajectory dataset from an InterSIM result."""

    _validate_params(params)
    omics: dict[OmicsLayer, pd.DataFrame] = {
        "methylation": intersim_result.methylation.copy(deep=True),
        "expression": intersim_result.expression.copy(deep=True),
        "proteomics": intersim_result.proteomics.copy(deep=True),
    }
    _validate_intersim_alignment(intersim_result, omics)

    stage_mapping = _build_stage_mapping(intersim_result.clusters)
    metadata = _build_sample_metadata(intersim_result, stage_mapping)
    n_stages = int(metadata["stage"].nunique())
    if params.trajectory_mode == "shape" and n_stages < 3:
        raise SemiSyntheticTrajectoryError("trajectory_mode='shape' requires at least three stages.")

    rng = np.random.default_rng(params.seed)
    metadata["group"] = _assign_groups_within_stages(metadata["stage"], params, rng)

    affected_features = _select_affected_features(omics, params, rng)
    effect_coefficients = _stage_effect_coefficients(params.trajectory_mode, n_stages)
    truth: dict[str, Any] = {
        "trajectory_mode": params.trajectory_mode,
        "group_effect_size": params.group_effect_size,
        "group_labels": list(params.group_labels),
        "group_ratio": params.group_ratio,
        "seed": params.seed,
        "stage_mapping": {str(k): v for k, v in stage_mapping.items()},
        "stage_assumption": "clusters-as-stages",
        "affected_features": affected_features,
        "effect_coefficients": effect_coefficients.tolist(),
        "effect_vectors": {},
        "intersim_metadata": intersim_result.metadata,
    }

    if params.trajectory_mode != "none" and params.group_effect_size != 0:
        for layer in _OMICS_LAYERS:
            effect_vector = _effect_vector_for_layer(
                layer=layer,
                columns=omics[layer].columns,
                affected_features=affected_features[layer],
                mode=params.trajectory_mode,
            )
            _apply_group_effect(
                matrix=omics[layer],
                metadata=metadata,
                params=params,
                effect_coefficients=effect_coefficients,
                effect_vector=effect_vector,
            )
            truth["effect_vectors"][layer] = {
                feature: float(effect_vector.loc[feature])
                for feature in affected_features[layer]
            }
    else:
        truth["effect_vectors"] = {layer: {} for layer in _OMICS_LAYERS}

    _validate_dataset_alignment(omics, metadata)
    return SemiSyntheticTrajectoryDataset(
        methylation=omics["methylation"],
        expression=omics["expression"],
        proteomics=omics["proteomics"],
        metadata=metadata,
        truth=truth,
    )


def generate_semisynthetic_trajectory_from_intersim(
    intersim_params: InterSIMParams,
    trajectory_params: SemiSyntheticTrajectoryParams,
    *,
    rscript: str = "Rscript",
    check_dependency: bool = True,
) -> SemiSyntheticTrajectoryDataset:
    """Invoke InterSIM and generate a semi-synthetic trajectory dataset."""

    intersim_result = run_intersim(
        intersim_params,
        rscript=rscript,
        check_dependency=check_dependency,
    )
    return generate_semisynthetic_trajectory(intersim_result, trajectory_params)


def _validate_params(params: SemiSyntheticTrajectoryParams) -> None:
    if params.trajectory_mode not in {"none", "translation", "magnitude", "orientation", "shape"}:
        raise SemiSyntheticTrajectoryError(f"Unknown trajectory_mode: {params.trajectory_mode}")
    if len(params.group_labels) != 2 or params.group_labels[0] == params.group_labels[1]:
        raise SemiSyntheticTrajectoryError("group_labels must contain two distinct labels.")
    if not (0 < params.group_ratio < 1):
        raise SemiSyntheticTrajectoryError("group_ratio must be between 0 and 1.")
    if params.group_effect_size < 0:
        raise SemiSyntheticTrajectoryError("group_effect_size must be non-negative.")


def _validate_intersim_alignment(intersim_result: InterSIMResult, omics: Mapping[OmicsLayer, pd.DataFrame]) -> None:
    expected = intersim_result.sample_ids.astype(str).tolist()
    clusters = intersim_result.clusters
    if clusters.shape[0] != len(expected):
        raise SemiSyntheticTrajectoryError("clusters must have the same number of rows as sample_ids.")
    if clusters.index.astype(str).tolist() != expected:
        raise SemiSyntheticTrajectoryError("clusters must use the same sample ID order as sample_ids.")
    for layer, matrix in omics.items():
        if matrix.index.astype(str).tolist() != expected:
            raise SemiSyntheticTrajectoryError(f"{layer} rows must match sample_ids.")


def _build_stage_mapping(clusters: pd.Series) -> dict[Any, int]:
    labels = pd.unique(clusters)
    sorted_labels = sorted(labels, key=_cluster_sort_key)
    return {label: idx for idx, label in enumerate(sorted_labels)}


def _cluster_sort_key(label: Any) -> tuple[int, float | str]:
    try:
        return (0, float(label))
    except (TypeError, ValueError):
        return (1, str(label))


def _build_sample_metadata(intersim_result: InterSIMResult, stage_mapping: Mapping[Any, int]) -> pd.DataFrame:
    metadata = pd.DataFrame(
        {
            "sample_id": intersim_result.sample_ids.astype(str),
            "cluster": intersim_result.clusters.to_numpy(),
        },
        index=intersim_result.sample_ids,
    )
    metadata["stage"] = intersim_result.clusters.map(stage_mapping).astype(int).to_numpy()
    return metadata


def _assign_groups_within_stages(
    stages: pd.Series,
    params: SemiSyntheticTrajectoryParams,
    rng: np.random.Generator,
) -> list[str]:
    groups = pd.Series(index=stages.index, dtype=object)
    first_label, second_label = params.group_labels
    for stage in sorted(pd.unique(stages)):
        idx = stages.index[stages == stage].to_numpy()
        n = len(idx)
        if n < 2:
            raise SemiSyntheticTrajectoryError(
                f"stage {stage} has {n} sample(s); at least 2 are required to assign both groups."
            )
        n_first = int(round(n * params.group_ratio))
        n_first = min(max(n_first, 1), n - 1)
        labels = np.array([first_label] * n_first + [second_label] * (n - n_first), dtype=object)
        rng.shuffle(labels)
        groups.loc[idx] = labels
    return groups.astype(str).tolist()


def _select_affected_features(
    omics: Mapping[OmicsLayer, pd.DataFrame],
    params: SemiSyntheticTrajectoryParams,
    rng: np.random.Generator,
) -> dict[OmicsLayer, list[str]]:
    explicit = params.affected_features or {}
    selected: dict[OmicsLayer, list[str]] = {}
    for layer in _OMICS_LAYERS:
        columns = omics[layer].columns.astype(str).tolist()
        explicit_features = explicit.get(layer)
        if explicit_features is not None:
            features = [str(feature) for feature in explicit_features]
            missing = sorted(set(features) - set(columns))
            if missing:
                raise SemiSyntheticTrajectoryError(f"affected_features[{layer!r}] contains unknown features: {missing}")
            selected[layer] = features
            continue

        proportion = _proportion_for_layer(params.prop_affected_features, layer)
        if not (0 <= proportion <= 1):
            raise SemiSyntheticTrajectoryError(f"prop_affected_features for {layer} must be between 0 and 1.")
        n_features = int(round(len(columns) * proportion))
        if proportion > 0 and n_features == 0:
            n_features = 1
        if n_features == 0:
            selected[layer] = []
            continue
        indices = rng.choice(len(columns), size=n_features, replace=False)
        selected[layer] = [columns[i] for i in sorted(indices)]
    return selected


def _proportion_for_layer(prop: float | Mapping[OmicsLayer, float], layer: OmicsLayer) -> float:
    if isinstance(prop, Mapping):
        return float(prop.get(layer, 0.0))
    return float(prop)


def _stage_effect_coefficients(mode: TrajectoryMode, n_stages: int) -> np.ndarray:
    if mode == "none":
        return np.zeros(n_stages, dtype=float)
    if mode == "translation":
        return np.ones(n_stages, dtype=float)
    if mode in {"magnitude", "orientation"}:
        return np.arange(n_stages, dtype=float)
    if mode == "shape":
        coeffs = np.zeros(n_stages, dtype=float)
        coeffs[n_stages // 2] = 1.0
        return coeffs
    raise SemiSyntheticTrajectoryError(f"Unknown trajectory_mode: {mode}")


def _effect_vector_for_layer(
    *,
    layer: OmicsLayer,
    columns: pd.Index,
    affected_features: Sequence[str],
    mode: TrajectoryMode,
) -> pd.Series:
    vector = pd.Series(0.0, index=columns.astype(str))
    if not affected_features:
        return vector
    signs = _feature_signs(len(affected_features), mode)
    for feature, sign in zip(affected_features, signs):
        vector.loc[str(feature)] = sign
    return vector


def _feature_signs(n_features: int, mode: TrajectoryMode) -> np.ndarray:
    signs = np.ones(n_features, dtype=float)
    if mode in {"orientation", "shape"} and n_features > 1:
        signs[1::2] = -1.0
    return signs


def _apply_group_effect(
    *,
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    params: SemiSyntheticTrajectoryParams,
    effect_coefficients: np.ndarray,
    effect_vector: pd.Series,
) -> None:
    target_group = params.group_labels[1]
    if not np.any(effect_vector.to_numpy() != 0):
        return
    for stage, coefficient in enumerate(effect_coefficients):
        if coefficient == 0:
            continue
        mask = (metadata["group"] == target_group) & (metadata["stage"] == stage)
        if not mask.any():
            continue
        sample_ids = metadata.loc[mask, "sample_id"].tolist()
        shift = params.group_effect_size * coefficient * effect_vector
        matrix.loc[sample_ids, shift.index] = matrix.loc[sample_ids, shift.index] + shift.to_numpy()


def _validate_dataset_alignment(omics: Mapping[OmicsLayer, pd.DataFrame], metadata: pd.DataFrame) -> None:
    expected = metadata["sample_id"].astype(str).tolist()
    for layer, matrix in omics.items():
        if matrix.index.astype(str).tolist() != expected:
            raise SemiSyntheticTrajectoryError(f"{layer} rows are not aligned to sample metadata.")
