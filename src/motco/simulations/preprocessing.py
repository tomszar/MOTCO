"""Shared pooled preprocessing for semi-synthetic integration and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from motco.simulations.generator import logit
from motco.simulations.semisynthetic import OmicsLayer, PopulationTrajectories, SemiSyntheticTrajectoryDataset

OMIC_LAYERS: tuple[OmicsLayer, ...] = ("methylation", "expression", "proteomics")
_SCALE_TOLERANCE = 1e-10


@dataclass(frozen=True)
class BlockScaler:
    """Fitted pooled location and scale for one aligned omic block."""

    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray


@dataclass(frozen=True)
class FittedOmicsPreprocessor:
    """Pooled per-feature scalers in canonical omic order."""

    scalers: Mapping[OmicsLayer, BlockScaler]
    methylation_units: str = "mvalue"

    def transform_dataset(self, dataset: SemiSyntheticTrajectoryDataset) -> dict[OmicsLayer, pd.DataFrame]:
        """Transform observed blocks using the fitted feature contract."""

        transformed: dict[OmicsLayer, pd.DataFrame] = {}
        for layer in OMIC_LAYERS:
            matrix = getattr(dataset, layer).astype(float)
            scaler = self.scalers[layer]
            _validate_features(layer, matrix.columns, scaler.feature_names)
            values = _integration_values(layer, matrix.to_numpy(dtype=float))
            transformed[layer] = pd.DataFrame(
                (values - scaler.mean) / scaler.scale,
                index=matrix.index.astype(str),
                columns=scaler.feature_names,
            )
        return transformed

    def transform_population(self, population: PopulationTrajectories) -> dict[OmicsLayer, pd.DataFrame]:
        """Transform analytic means with the same observed-fitted scalers."""

        transformed: dict[OmicsLayer, pd.DataFrame] = {}
        for layer in OMIC_LAYERS:
            matrix = population.layers[layer].astype(float)
            scaler = self.scalers[layer]
            _validate_features(layer, matrix.columns, scaler.feature_names)
            transformed[layer] = pd.DataFrame(
                (matrix.to_numpy(dtype=float) - scaler.mean) / scaler.scale,
                index=matrix.index,
                columns=scaler.feature_names,
            )
        return transformed


def fit_omics_preprocessor(dataset: SemiSyntheticTrajectoryDataset) -> FittedOmicsPreprocessor:
    """Fit pooled block-wise scalers on the observed dataset."""

    scalers: dict[OmicsLayer, BlockScaler] = {}
    for layer in OMIC_LAYERS:
        matrix = getattr(dataset, layer).astype(float)
        values = _integration_values(layer, matrix.to_numpy(dtype=float))
        scale = values.std(axis=0)
        scale[scale < _SCALE_TOLERANCE] = 1.0
        scalers[layer] = BlockScaler(
            feature_names=tuple(matrix.columns.astype(str)),
            mean=values.mean(axis=0),
            scale=scale,
        )
    return FittedOmicsPreprocessor(scalers=scalers)


def concatenate_blocks(blocks: Mapping[OmicsLayer, pd.DataFrame]) -> pd.DataFrame:
    """Concatenate aligned standardized blocks with collision-proof names."""

    frames = []
    for layer in OMIC_LAYERS:
        matrix = blocks[layer]
        renamed = matrix.copy()
        renamed.columns = [f"{layer}__{column}" for column in matrix.columns.astype(str)]
        frames.append(renamed)
    return pd.concat(frames, axis=1)


def _integration_values(layer: OmicsLayer, values: np.ndarray) -> np.ndarray:
    return logit(values) if layer == "methylation" else values


def _validate_features(layer: str, columns: pd.Index, expected: tuple[str, ...]) -> None:
    actual = tuple(columns.astype(str))
    if actual != expected:
        raise ValueError(f"{layer} feature order does not match the fitted preprocessing artifact.")
