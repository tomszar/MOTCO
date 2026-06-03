"""Evaluation harness for semi-synthetic trajectory datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Literal, Mapping

import numpy as np
import pandas as pd

from motco.simulations.semisynthetic import SemiSyntheticTrajectoryDataset
from motco.stats.design import build_ls_means, get_model_matrix
from motco.stats.permutation import RRPP
from motco.stats.snf import SNF, get_affinity_matrix, get_spectral
from motco.stats.trajectory import estimate_difference

IntegrationMethod = Literal["concat", "snf"]

_OMICS_ATTRS: tuple[str, ...] = ("methylation", "expression", "proteomics")


class SimulationEvaluationError(ValueError):
    """Raised when simulation evaluation inputs or parameters are invalid."""


@dataclass(frozen=True)
class SimulationEvaluationParams:
    """Parameters for evaluating one semi-synthetic trajectory dataset."""

    integration_method: IntegrationMethod = "concat"
    integration_params: Mapping[str, Any] = field(default_factory=dict)
    permutations: int = 0
    n_jobs: int | None = 1
    seed: int | None = None
    progress: bool = False
    include_null_distributions: bool = False
    group_col: str = "group"
    stage_col: str = "stage"


@dataclass(frozen=True)
class LatentIntegrationResult:
    """Integrated latent/outcome matrix and metadata."""

    matrix: pd.DataFrame
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SimulationTrajectoryDesign:
    """Trajectory design objects derived from sample metadata."""

    model_full: np.ndarray
    model_reduced: np.ndarray
    ls_means: np.ndarray
    contrast: list[list[int]]
    group_levels: list[str]
    stage_levels: list[str]


@dataclass(frozen=True)
class SimulationEvaluationResult:
    """Result from evaluating one semi-synthetic trajectory dataset."""

    observed_deltas: np.ndarray
    observed_angles: np.ndarray
    observed_shapes: np.ndarray
    pair_statistics: dict[str, float]
    p_values: dict[str, float]
    latent_matrix_metadata: dict[str, Any]
    truth_metadata: dict[str, Any]
    runtime_metadata: dict[str, Any]
    evaluation_params: SimulationEvaluationParams
    group_levels: list[str]
    stage_levels: list[str]
    contrast: list[list[int]]
    null_distributions: dict[str, list[float]] | None = None


def evaluate_semisynthetic_trajectory(
    dataset: SemiSyntheticTrajectoryDataset,
    params: SimulationEvaluationParams | None = None,
) -> SimulationEvaluationResult:
    """Evaluate one semi-synthetic trajectory dataset through MOTCO routines."""

    params = params or SimulationEvaluationParams()
    _validate_evaluation_params(params)
    start = perf_counter()

    latent = integrate_semisynthetic_dataset(dataset, params)
    design = build_simulation_trajectory_design(
        dataset.metadata,
        group_col=params.group_col,
        stage_col=params.stage_col,
    )

    observed_deltas, observed_angles, observed_shapes = estimate_difference(
        Y=latent.matrix,
        model_matrix=design.model_full,
        LS_means=design.ls_means,
        contrast=design.contrast,
    )
    shape_available = len(design.stage_levels) >= 3
    pair_statistics = _extract_pair_statistics(
        observed_deltas,
        observed_angles,
        observed_shapes,
        shape_available=shape_available,
    )

    p_values: dict[str, float] = {}
    null_distributions: dict[str, list[float]] | None = None
    if params.permutations > 0:
        dist_delta, dist_angle, dist_shape = RRPP(
            latent.matrix,
            design.model_full,
            design.model_reduced,
            design.ls_means,
            design.contrast,
            permutations=params.permutations,
            n_jobs=params.n_jobs,
            progress=params.progress,
            seed=params.seed,
        )
        null_values = _extract_null_distributions(
            dist_delta,
            dist_angle,
            dist_shape,
            shape_available=shape_available,
        )
        p_values = {
            statistic: _empirical_p_value(null_values[statistic], observed)
            for statistic, observed in pair_statistics.items()
            if statistic in null_values and np.isfinite(observed)
        }
        if params.include_null_distributions:
            null_distributions = null_values

    runtime_seconds = perf_counter() - start
    runtime_metadata = {
        "runtime_seconds": runtime_seconds,
        "permutations": params.permutations,
        "n_jobs": params.n_jobs,
        "seed": params.seed,
        "progress": params.progress,
        "shape_available": shape_available,
    }

    return SimulationEvaluationResult(
        observed_deltas=observed_deltas,
        observed_angles=observed_angles,
        observed_shapes=observed_shapes,
        pair_statistics=pair_statistics,
        p_values=p_values,
        latent_matrix_metadata=latent.metadata,
        # Drop the raw indicator arrays — they are not JSON-serializable for the
        # study's JSONL persistence; the per-stage/group counts and transform
        # notes (which are) are retained for downstream characterization.
        truth_metadata={k: v for k, v in dataset.truth.items() if k != "indicators"},
        runtime_metadata=runtime_metadata,
        evaluation_params=params,
        group_levels=design.group_levels,
        stage_levels=design.stage_levels,
        contrast=design.contrast,
        null_distributions=null_distributions,
    )


def integrate_semisynthetic_dataset(
    dataset: SemiSyntheticTrajectoryDataset,
    params: SimulationEvaluationParams,
) -> LatentIntegrationResult:
    """Create a latent/outcome matrix from aligned omics layers."""

    _validate_dataset(dataset, params.group_col, params.stage_col)
    method = params.integration_method
    if method == "concat":
        return _concat_integration(dataset, params.integration_params)
    if method == "snf":
        return _snf_integration(dataset, params.integration_params)
    raise SimulationEvaluationError(f"Unsupported integration_method: {method!r}.")


def build_simulation_trajectory_design(
    metadata: pd.DataFrame,
    *,
    group_col: str = "group",
    stage_col: str = "stage",
) -> SimulationTrajectoryDesign:
    """Build full/reduced model matrices, LS means, and two-group contrast."""

    _validate_metadata(metadata, group_col, stage_col)
    design_frame = metadata[[group_col, stage_col]].reset_index(drop=True).copy()
    design_frame[group_col] = design_frame[group_col].astype(str)
    design_frame[stage_col] = design_frame[stage_col].astype(str)
    group_levels = sorted(pd.unique(design_frame[group_col]).tolist())
    stage_levels = sorted(pd.unique(design_frame[stage_col]).tolist())
    if len(group_levels) != 2:
        raise SimulationEvaluationError(f"Expected exactly two groups, found {len(group_levels)}: {group_levels}.")
    if len(stage_levels) < 2:
        raise SimulationEvaluationError(
            f"Expected at least two stages, found {len(stage_levels)}: {stage_levels}."
        )
    _validate_group_stage_combinations(design_frame, group_col, stage_col, group_levels, stage_levels)

    model_full = get_model_matrix(design_frame, group_col=group_col, level_col=stage_col, full=True)
    model_reduced = get_model_matrix(design_frame, group_col=group_col, level_col=stage_col, full=False)
    ls_means = build_ls_means(group_levels, stage_levels, full=True)
    n_stages = len(stage_levels)
    contrast = [[group_index * n_stages + stage_index for stage_index in range(n_stages)] for group_index in range(2)]
    return SimulationTrajectoryDesign(
        model_full=model_full,
        model_reduced=model_reduced,
        ls_means=ls_means,
        contrast=contrast,
        group_levels=group_levels,
        stage_levels=stage_levels,
    )


def _validate_evaluation_params(params: SimulationEvaluationParams) -> None:
    if params.integration_method not in {"concat", "snf"}:
        raise SimulationEvaluationError(f"Unsupported integration_method: {params.integration_method!r}.")
    if params.permutations < 0:
        raise SimulationEvaluationError("permutations must be greater than or equal to 0.")
    if params.n_jobs == 0:
        raise SimulationEvaluationError("n_jobs must be None, -1, or a non-zero integer.")


def _validate_dataset(dataset: SemiSyntheticTrajectoryDataset, group_col: str, stage_col: str) -> None:
    _validate_metadata(dataset.metadata, group_col, stage_col)
    if "sample_id" not in dataset.metadata.columns:
        raise SimulationEvaluationError("metadata must contain required column 'sample_id'.")
    expected = dataset.metadata["sample_id"].astype(str).tolist()
    if len(expected) == 0:
        raise SimulationEvaluationError("metadata must contain at least one sample.")
    for layer in _OMICS_ATTRS:
        matrix = getattr(dataset, layer)
        if matrix.shape[0] != len(expected):
            raise SimulationEvaluationError(f"{layer} rows must match metadata rows.")
        if matrix.index.astype(str).tolist() != expected:
            raise SimulationEvaluationError(f"{layer} rows are not aligned to metadata sample_id order.")
        if matrix.shape[1] == 0:
            raise SimulationEvaluationError(f"{layer} must contain at least one feature.")
        values = matrix.to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise SimulationEvaluationError(f"{layer} contains NaN or Inf values.")


def _validate_metadata(metadata: pd.DataFrame, group_col: str, stage_col: str) -> None:
    required = [group_col, stage_col]
    missing = [column for column in required if column not in metadata.columns]
    if missing:
        raise SimulationEvaluationError(f"metadata is missing required column(s): {missing}.")


def _validate_group_stage_combinations(
    metadata: pd.DataFrame,
    group_col: str,
    stage_col: str,
    group_levels: list[str],
    stage_levels: list[str],
) -> None:
    present = set(zip(metadata[group_col].astype(str), metadata[stage_col].astype(str)))
    missing = [(group, stage) for group in group_levels for stage in stage_levels if (group, stage) not in present]
    if missing:
        raise SimulationEvaluationError(f"metadata is missing group/stage combination(s): {missing}.")


def _concat_integration(
    dataset: SemiSyntheticTrajectoryDataset,
    integration_params: Mapping[str, Any],
) -> LatentIntegrationResult:
    standardize = bool(integration_params.get("standardize", True))
    frames: list[pd.DataFrame] = []
    layer_feature_counts: dict[str, int] = {}
    for layer in _OMICS_ATTRS:
        matrix = getattr(dataset, layer).astype(float)
        layer_feature_counts[layer] = int(matrix.shape[1])
        values = matrix.to_numpy(dtype=float)
        if standardize:
            mean = values.mean(axis=0, keepdims=True)
            std = values.std(axis=0, keepdims=True)
            std[std == 0.0] = 1.0
            values = (values - mean) / std
        columns = [f"{layer}__{column}" for column in matrix.columns.astype(str)]
        frames.append(pd.DataFrame(values, index=matrix.index.astype(str), columns=columns))
    latent = pd.concat(frames, axis=1)
    return LatentIntegrationResult(
        matrix=latent,
        metadata={
            "integration_method": "concat",
            "integration_params": {"standardize": standardize},
            "shape": tuple(latent.shape),
            "n_samples": int(latent.shape[0]),
            "n_features": int(latent.shape[1]),
            "layer_feature_counts": layer_feature_counts,
        },
    )


def _snf_integration(
    dataset: SemiSyntheticTrajectoryDataset,
    integration_params: Mapping[str, Any],
) -> LatentIntegrationResult:
    n_samples = int(dataset.metadata.shape[0])
    K = _bounded_int_param(integration_params, "K", default=min(20, n_samples - 1), minimum=1, maximum=n_samples - 1)
    k = _bounded_int_param(integration_params, "k", default=min(20, n_samples - 1), minimum=1, maximum=n_samples - 1)
    t = _bounded_int_param(integration_params, "t", default=20, minimum=1)
    n_components = _bounded_int_param(
        integration_params,
        "spectral_components",
        default=min(10, n_samples - 1),
        minimum=1,
        maximum=n_samples - 1,
    )
    eps = float(integration_params.get("eps", 0.5))
    if eps <= 0:
        raise SimulationEvaluationError("SNF integration parameter 'eps' must be positive.")

    layers = [getattr(dataset, layer).to_numpy(dtype=float) for layer in _OMICS_ATTRS]
    affinities = get_affinity_matrix(layers, K=K, eps=eps)
    fused = SNF(affinities, k=k, t=t)
    embedding = get_spectral(fused, n_components=n_components)
    latent = pd.DataFrame(
        embedding,
        index=dataset.metadata["sample_id"].astype(str).tolist(),
        columns=[f"snf_{i}" for i in range(embedding.shape[1])],
    )
    return LatentIntegrationResult(
        matrix=latent,
        metadata={
            "integration_method": "snf",
            "integration_params": {
                "K": K,
                "eps": eps,
                "k": k,
                "t": t,
                "spectral_components": n_components,
            },
            "shape": tuple(latent.shape),
            "n_samples": int(latent.shape[0]),
            "n_features": int(latent.shape[1]),
            "fused_shape": tuple(fused.shape),
        },
    )


def _bounded_int_param(
    params: Mapping[str, Any],
    key: str,
    *,
    default: int,
    minimum: int,
    maximum: int | None = None,
) -> int:
    value = int(params.get(key, default))
    if value < minimum:
        raise SimulationEvaluationError(f"SNF integration parameter {key!r} must be at least {minimum}.")
    if maximum is not None and value > maximum:
        raise SimulationEvaluationError(f"SNF integration parameter {key!r} must be at most {maximum}.")
    return value


def _extract_pair_statistics(
    deltas: np.ndarray,
    angles: np.ndarray,
    shapes: np.ndarray,
    *,
    shape_available: bool,
) -> dict[str, float]:
    stats = {
        "delta": float(deltas[0, 1]),
        "angle": float(angles[0, 1]),
        "shape": float(shapes[0, 1]) if shape_available else float("nan"),
    }
    return stats


def _extract_null_distributions(
    dist_delta: list[np.ndarray],
    dist_angle: list[np.ndarray],
    dist_shape: list[np.ndarray],
    *,
    shape_available: bool,
) -> dict[str, list[float]]:
    out = {
        "delta": [float(matrix[0, 1]) for matrix in dist_delta],
        "angle": [float(matrix[0, 1]) for matrix in dist_angle],
    }
    if shape_available:
        out["shape"] = [float(matrix[0, 1]) for matrix in dist_shape]
    return out


def _empirical_p_value(null_values: list[float], observed: float) -> float:
    values = np.asarray(null_values, dtype=float)
    return float((1.0 + np.sum(values >= observed)) / (1.0 + values.size))
