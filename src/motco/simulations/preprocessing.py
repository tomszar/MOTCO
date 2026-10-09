"""Shared pooled preprocessing for semi-synthetic integration and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, cast

import numpy as np
import pandas as pd

from motco.simulations.generator import logit
from motco.simulations.semisynthetic import OmicsLayer, PopulationTrajectories, SemiSyntheticTrajectoryDataset

OMIC_LAYERS: tuple[OmicsLayer, ...] = ("methylation", "expression", "proteomics")
_SCALE_TOLERANCE = 1e-10


def selected_layers(integration_params: Mapping[str, Any] | None) -> tuple[OmicsLayer, ...]:
    """Resolve ``integration_params["layers"]`` to canonical omic order.

    Absent (or ``None``) means every layer. A present selection must be a
    non-empty, duplicate-free subset of :data:`OMIC_LAYERS`.
    """

    raw = None if integration_params is None else integration_params.get("layers")
    if raw is None:
        return OMIC_LAYERS
    if isinstance(raw, str | bytes) or not isinstance(raw, Sequence):
        raise ValueError(f"integration_params['layers'] must be a list of layer names from {list(OMIC_LAYERS)}.")
    names = [str(name) for name in raw]
    if not names:
        raise ValueError(f"integration_params['layers'] must not be empty; allowed layers: {list(OMIC_LAYERS)}.")
    unknown = sorted(set(names) - set(OMIC_LAYERS))
    if unknown:
        raise ValueError(
            f"integration_params['layers'] names unknown layer(s) {unknown}; allowed layers: {list(OMIC_LAYERS)}."
        )
    if len(set(names)) != len(names):
        raise ValueError(
            f"integration_params['layers'] repeats a layer: {names}; allowed layers: {list(OMIC_LAYERS)}."
        )
    return tuple(cast(OmicsLayer, layer) for layer in OMIC_LAYERS if layer in names)


@dataclass(frozen=True)
class BlockScaler:
    """Fitted pooled location and scale for one aligned omic block."""

    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray


@dataclass(frozen=True)
class FittedOmicsPreprocessor:
    """Pooled per-feature scalers in canonical omic order.

    ``scalers`` holds exactly the layers the preprocessor was fitted on;
    :attr:`layers` lists them in canonical order, and every transform covers
    those layers only.
    """

    scalers: Mapping[OmicsLayer, BlockScaler]
    methylation_units: str = "mvalue"

    @property
    def layers(self) -> tuple[OmicsLayer, ...]:
        return tuple(layer for layer in OMIC_LAYERS if layer in self.scalers)

    def require_layers(self, layers: Sequence[OmicsLayer]) -> None:
        """Raise unless the preprocessor was fitted on exactly ``layers``."""

        if tuple(layers) != self.layers:
            raise ValueError(
                f"Preprocessor was fitted on layers {list(self.layers)}, not the requested {list(layers)}."
            )

    def transform_dataset(self, dataset: SemiSyntheticTrajectoryDataset) -> dict[OmicsLayer, pd.DataFrame]:
        """Transform observed blocks using the fitted feature contract."""

        transformed: dict[OmicsLayer, pd.DataFrame] = {}
        for layer in self.layers:
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
        for layer in self.layers:
            matrix = population.layers[layer].astype(float)
            scaler = self.scalers[layer]
            _validate_features(layer, matrix.columns, scaler.feature_names)
            transformed[layer] = pd.DataFrame(
                (matrix.to_numpy(dtype=float) - scaler.mean) / scaler.scale,
                index=matrix.index,
                columns=scaler.feature_names,
            )
        return transformed


def fit_omics_preprocessor(
    dataset: SemiSyntheticTrajectoryDataset,
    layers: Sequence[OmicsLayer] = OMIC_LAYERS,
) -> FittedOmicsPreprocessor:
    """Fit pooled block-wise scalers on the observed dataset's ``layers``."""

    scalers: dict[OmicsLayer, BlockScaler] = {}
    for layer in layers:
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
    """Concatenate aligned standardized blocks with collision-proof names.

    Blocks are taken in canonical omic order; only the layers present in
    ``blocks`` are concatenated.
    """

    frames = []
    for layer in OMIC_LAYERS:
        if layer not in blocks:
            continue
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
