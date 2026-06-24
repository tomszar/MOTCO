"""Multi-block concatenation test bed for trajectory recovery (Rung 3).

Adds exactly one factor on top of the Rung-2 clean floor: multiple heterogeneous
omic blocks concatenated after per-block z-scoring — the production ``concat``
transform.  The known 2-stage geometry (``none``/``magnitude``/``orientation``) is
injected into a single **anchor block** (M-value space, no ``rev.logit`` — the
Rung-1/2 clean frame), and one or two **nuisance blocks** of configurable
dimensionality and exchangeable-correlation structure are appended.  All blocks are
independently z-scored and column-concatenated before PCA, exactly as in
``evaluation.py:_concat_integration``.

When the nuisance blocks dominate the joint feature matrix (high ``dim_ratio``),
the top principal components may tilt away from the anchor block's geometry-carrying
subspace.  This rung measures whether that tilt manufactures magnitude→orientation
cross-talk on an otherwise clean linear problem.

The estimator path (``get_model_matrix``/``build_ls_means``/``estimate_difference``,
2-group × 2-stage design, two-group contrast) is identical to Rungs 0–2.
Cross-talk is read against the per-configuration ``none`` null and against the
``n_nuisance_blocks = 0`` single-block floor (which reproduces the Rung-2
``standardize`` baseline exactly).

This is Rung 3 of a ladder:
- Rung 0 (``linear_recovery``)      — proved the linear floor is clean.
- Rung 1 (``methylation_recovery``) — showed ``rev.logit`` is inverted by M-value
                                       integration; not a cross-talk source.
- Rung 2 (``projector_recovery``)   — showed per-feature standardization is clean
                                       on a single block; SNF leaks.
- Rung 3 (this module)              — adds multiple heterogeneous blocks and asks
                                       whether block-size imbalance manufactures
                                       cross-talk.

References
----------
openspec/changes/rung3-multiblock-concatenation/design.md
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.decomposition import PCA

from motco.simulations.linear_recovery import LinearRecoveryParams
from motco.simulations.linear_recovery import generate_dataset as _linear_generate_dataset
from motco.stats.design import build_ls_means, get_model_matrix
from motco.stats.trajectory import estimate_difference

__all__ = [
    "MultiblockRecoveryError",
    "MultiblockRecoveryParams",
    "MultiblockRecoveryDataset",
    "generate_dataset",
    "build_joint_matrix",
    "decompose_block_weight",
    "project_and_measure",
    "run_block_comparison",
    "run_dim_ratio_sweep",
    "run_rho_sweep",
    "run_effect_size_sweep",
    "run_block_weight_curve",
    "plot_dim_ratio_sweep",
    "plot_block_weight_curve",
]


class MultiblockRecoveryError(ValueError):
    """Raised on invalid ``MultiblockRecoveryParams``."""


_MANIPULATIONS: list[Literal["none", "magnitude", "orientation"]] = [
    "none",
    "magnitude",
    "orientation",
]
_DEFAULT_DIM_RATIOS: list[float] = [0.5, 1.0, 2.0, 5.0, 10.0]
_DEFAULT_RHO_VALUES: list[float] = [0.0, 0.3, 0.7]


# ---------------------------------------------------------------------------
# Parameters and dataset
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MultiblockRecoveryParams:
    """Frozen configuration for a Rung-3 multi-block concatenation run.

    The anchor block carries the known trajectory geometry (identical to
    Rungs 0–2).  Each nuisance block is drawn independently from a zero-mean
    MVN with exchangeable covariance
    ``Σ = σ²[(1 − ρ)I + ρ·11ᵀ]`` (``rho_nuisance = ρ``).

    Parameters
    ----------
    seed:
        RNG seed for deterministic generation.
    n_features_anchor:
        Feature dimensionality of the anchor (geometry-carrying) block.
    n_samples_per_cell:
        Samples drawn per (group, stage) cell; total = 4 × this.
    noise_scale:
        Standard deviation ``σ`` applied to all blocks (anchor + nuisance).
    signal_scale:
        ‖a_feat‖ — the magnitude of group A's feature-space step in the anchor.
    manipulation:
        Geometric transform applied to group B's step vector.
    scale_c:
        Magnitude scale for the ``"magnitude"`` manipulation (step_B = c · step_A).
    angle_theta:
        Rotation angle in degrees for the ``"orientation"`` manipulation.
    n_nuisance_blocks:
        Number of nuisance blocks appended (0 = single-block Rung-2 baseline).
    dim_ratio:
        Nuisance-block dimensionality relative to the anchor:
        ``p_nuisance = max(1, round(dim_ratio × n_features_anchor))``.
        Only meaningful when ``n_nuisance_blocks > 0``.
    rho_nuisance:
        Exchangeable correlation coefficient for the nuisance blocks (``0`` =
        independent features; ``< 1``).  The anchor block always has isotropic
        noise.
    n_components:
        Number of PCA components retained from the joint matrix for measurement.
    """

    seed: int = 0
    n_features_anchor: int = 50
    n_samples_per_cell: int = 40
    noise_scale: float = 1.0
    signal_scale: float = 5.0
    manipulation: Literal["none", "magnitude", "orientation"] = "none"
    scale_c: float = 2.0
    angle_theta: float = 45.0
    n_nuisance_blocks: int = 1
    dim_ratio: float = 1.0
    rho_nuisance: float = 0.0
    n_components: int = 10

    def as_linear(self) -> LinearRecoveryParams:
        """Project to Rung-0 params for step-construction reuse."""
        return LinearRecoveryParams(
            seed=self.seed,
            n_features=self.n_features_anchor,
            n_samples_per_cell=self.n_samples_per_cell,
            noise_scale=self.noise_scale,
            signal_scale=self.signal_scale,
            manipulation=self.manipulation,
            scale_c=self.scale_c,
            angle_theta=self.angle_theta,
            n_components=self.n_components,
        )

    @property
    def n_features_nuisance(self) -> int:
        """Feature count per nuisance block (0 when no nuisance blocks)."""
        if self.n_nuisance_blocks == 0:
            return 0
        return max(1, round(self.dim_ratio * self.n_features_anchor))

    @property
    def n_features_total(self) -> int:
        """Total feature count in the joint (concatenated) matrix."""
        return self.n_features_anchor + self.n_nuisance_blocks * self.n_features_nuisance

    @property
    def n_samples(self) -> int:
        """Total number of samples across all (group, stage) cells."""
        return 4 * self.n_samples_per_cell


@dataclass
class MultiblockRecoveryDataset:
    """Output of :func:`generate_dataset`.

    Attributes
    ----------
    X_anchor:
        Anchor feature matrix (n_samples × n_features_anchor) in M-value space.
    X_nuisance:
        List of nuisance feature matrices (n_samples × p_nuisance each), one
        per nuisance block.  Empty when ``n_nuisance_blocks = 0``.
    metadata:
        Sample metadata with ``"group"`` (A/B) and ``"stage"`` (0/1) columns,
        row-aligned with all feature matrices.
    step_A:
        Ground-truth anchor-block step for group A (μ_{A,1} − μ_{A,0}).
    step_B:
        Ground-truth anchor-block step for group B.
    """

    X_anchor: np.ndarray
    X_nuisance: list[np.ndarray]
    metadata: pd.DataFrame
    step_A: np.ndarray
    step_B: np.ndarray


# ---------------------------------------------------------------------------
# Validation and generation
# ---------------------------------------------------------------------------


def _validate(p: MultiblockRecoveryParams) -> None:
    if p.n_samples_per_cell < 2:
        raise MultiblockRecoveryError("n_samples_per_cell must be >= 2")
    if p.n_components < 2:
        raise MultiblockRecoveryError("n_components must be >= 2")
    if p.n_components > p.n_features_total:
        raise MultiblockRecoveryError(
            f"n_components ({p.n_components}) must be <= n_features_total "
            f"({p.n_features_total})"
        )
    if p.n_components >= p.n_samples:
        raise MultiblockRecoveryError(
            f"n_components ({p.n_components}) must be < n_samples ({p.n_samples})"
        )
    if p.noise_scale <= 0:
        raise MultiblockRecoveryError("noise_scale must be > 0")
    if p.signal_scale <= 0:
        raise MultiblockRecoveryError("signal_scale must be > 0")
    if p.n_nuisance_blocks < 0:
        raise MultiblockRecoveryError("n_nuisance_blocks must be >= 0")
    if p.n_nuisance_blocks > 0 and p.dim_ratio <= 0:
        raise MultiblockRecoveryError(
            "dim_ratio must be > 0 when n_nuisance_blocks > 0"
        )
    if not (0.0 <= p.rho_nuisance < 1.0):
        raise MultiblockRecoveryError("rho_nuisance must be in [0, 1)")
    if p.manipulation not in ("none", "magnitude", "orientation"):
        raise MultiblockRecoveryError(
            f"manipulation must be 'none', 'magnitude', or 'orientation'; "
            f"got {p.manipulation!r}"
        )


def generate_dataset(params: MultiblockRecoveryParams) -> MultiblockRecoveryDataset:
    """Generate anchor + nuisance blocks with known anchor-block trajectory geometry.

    The anchor block step vectors are constructed by the Rung-0 generator
    (reused unchanged), so the injected geometry is identical to Rungs 0–2.
    Anchor-block samples are drawn from ``N(μ_{g,s}, σ²I)``; nuisance blocks
    from ``N(0, σ²[(1−ρ)I + ρ·11ᵀ])`` with no group or stage structure.

    A single RNG seeded by ``params.seed`` drives all sample generation in order:
    anchor cells first (matching the Rung-2 ``standardize`` baseline when
    ``n_nuisance_blocks = 0``), then each nuisance block in sequence.

    Parameters
    ----------
    params:
        Generation configuration.
    """
    _validate(params)

    lin = _linear_generate_dataset(params.as_linear())
    step_A, step_B = lin.step_A, lin.step_B

    p_a = params.n_features_anchor
    n = params.n_samples_per_cell
    mu: dict[tuple[str, str], np.ndarray] = {
        ("A", "0"): np.zeros(p_a),
        ("A", "1"): step_A,
        ("B", "0"): np.zeros(p_a),
        ("B", "1"): step_B,
    }

    rng = np.random.default_rng(params.seed)

    # Anchor block — isotropic noise, identical to Rung-2 standardize baseline.
    anchor_rows: list[np.ndarray] = []
    meta_rows: list[tuple[str, str]] = []
    for group in ("A", "B"):
        for stage in ("0", "1"):
            noise = rng.standard_normal((n, p_a)) * params.noise_scale
            anchor_rows.append(mu[(group, stage)] + noise)
            meta_rows.extend([(group, stage)] * n)
    X_anchor = np.vstack(anchor_rows)

    # Nuisance blocks — zero-mean, exchangeable correlation, no group/stage signal.
    n_total = params.n_samples
    p_nuis = params.n_features_nuisance
    X_nuisance: list[np.ndarray] = []
    for _ in range(params.n_nuisance_blocks):
        rho = params.rho_nuisance
        if rho > 0:
            # Factor model: x = sqrt(ρ)·σ·f + sqrt(1−ρ)·σ·e
            # where f ~ N(0,1) is a common factor (n_total,) and
            # e ~ N(0, I) is independent noise (n_total, p_nuis).
            common = rng.standard_normal(n_total) * (
                params.noise_scale * np.sqrt(rho)
            )
            indep = rng.standard_normal((n_total, p_nuis)) * (
                params.noise_scale * np.sqrt(1.0 - rho)
            )
            X_nuisance.append(common[:, None] + indep)
        else:
            X_nuisance.append(
                rng.standard_normal((n_total, p_nuis)) * params.noise_scale
            )

    metadata = pd.DataFrame(meta_rows, columns=["group", "stage"])
    return MultiblockRecoveryDataset(
        X_anchor=X_anchor,
        X_nuisance=X_nuisance,
        metadata=metadata,
        step_A=step_A,
        step_B=step_B,
    )


# ---------------------------------------------------------------------------
# Block concatenation and projection
# ---------------------------------------------------------------------------


def build_joint_matrix(dataset: MultiblockRecoveryDataset) -> tuple[np.ndarray, int]:
    """Per-block z-score and concatenate all blocks.

    Each block is standardized independently (``std == 0 → 1``, matching
    ``evaluation.py:_concat_integration``) then column-concatenated.

    Parameters
    ----------
    dataset:
        Output of :func:`generate_dataset`.

    Returns
    -------
    X_joint : np.ndarray
        Joint standardised feature matrix, shape
        ``(n_samples, n_features_total)``.
    p_anchor : int
        Number of anchor-block columns (first columns of ``X_joint``).
    """
    blocks = [dataset.X_anchor] + dataset.X_nuisance
    standardised: list[np.ndarray] = []
    for X in blocks:
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True)
        std[std == 0.0] = 1.0
        standardised.append((X - mean) / std)
    return np.hstack(standardised), dataset.X_anchor.shape[1]


def decompose_block_weight(
    X_joint: np.ndarray,
    p_anchor: int,
    n_components: int,
) -> tuple[float, np.ndarray]:
    """Anchor block's fraction of variance in the top-k PCA components.

    PCA is fit on ``X_joint`` and the loading matrix
    ``V = pca.components_.T`` (shape ``p_total × k``) is decomposed by block.
    Each column of ``V`` is unit-norm (sklearn convention), so the Frobenius
    norm of the full loading matrix equals ``k``.

    Parameters
    ----------
    X_joint:
        Joint standardised matrix from :func:`build_joint_matrix`.
    p_anchor:
        Number of leading columns that belong to the anchor block.
    n_components:
        Number of PCA components to retain.

    Returns
    -------
    w_anchor : float
        Overall anchor-block loading fraction
        ``‖V[:p_anchor, :]‖²_F / ‖V‖²_F ∈ (0, 1)``.
        Approaches ``p_anchor / p_total`` for uncorrelated blocks of the same
        noise scale; diverges from it when one block dominates.
    per_component : np.ndarray, shape (k,)
        Per-component anchor fraction
        ``‖V[:p_anchor, j]‖² / ‖V[:, j]‖² = ‖V[:p_anchor, j]‖²``
        (denominator is 1 by construction).
    """
    pca = PCA(n_components=n_components)
    pca.fit(X_joint)
    V = pca.components_.T  # (p_total, k); columns are unit-norm
    anchor_sq = float((V[:p_anchor, :] ** 2).sum())
    total_sq = float((V**2).sum())  # = k
    w_anchor = anchor_sq / total_sq
    per_component = (V[:p_anchor, :] ** 2).sum(axis=0)  # (k,)
    return w_anchor, per_component


def _measure_projected(
    Y: np.ndarray,
    metadata: pd.DataFrame,
    n_components: int,
) -> tuple[float, float]:
    """Measure group-A-vs-B delta/angle on an (n_samples × k) latent matrix."""
    Y_df = pd.DataFrame(
        Y,
        columns=[f"comp{i + 1}" for i in range(n_components)],
        index=metadata.index,
    )
    model_matrix = get_model_matrix(
        metadata, group_col="group", level_col="stage", full=True
    )
    g_levels = sorted(metadata["group"].astype(str).unique().tolist())
    l_levels = sorted(metadata["stage"].astype(str).unique().tolist())
    ls_means = build_ls_means(g_levels, l_levels, full=True)
    n_l = len(l_levels)
    contrast = [list(range(i * n_l, (i + 1) * n_l)) for i in range(len(g_levels))]
    deltas, angles, _ = estimate_difference(Y_df, model_matrix, ls_means, contrast)
    return float(deltas[0, 1]), float(angles[0, 1])


def project_and_measure(
    dataset: MultiblockRecoveryDataset,
    params: MultiblockRecoveryParams,
) -> tuple[float, float, np.ndarray]:
    """Build the joint matrix, fit PCA, and measure delta/angle.

    Returns
    -------
    delta : float
        Group-A vs Group-B magnitude difference in the joint PCA space.
    angle : float
        Group-A vs Group-B direction difference (degrees).
    Y : np.ndarray
        Projected outcome matrix (n_samples × n_components).
    """
    X_joint, _ = build_joint_matrix(dataset)
    pca = PCA(n_components=params.n_components)
    Y = pca.fit_transform(X_joint)
    delta, angle = _measure_projected(Y, dataset.metadata, params.n_components)
    return delta, angle, Y


# ---------------------------------------------------------------------------
# Drivers / sweeps
# ---------------------------------------------------------------------------


def _measure_over_seeds(
    base_params: MultiblockRecoveryParams,
    manip: Literal["none", "magnitude", "orientation"],
    seeds: list[int],
    **overrides: Any,
) -> tuple[list[float], list[float]]:
    deltas: list[float] = []
    angles: list[float] = []
    for seed in seeds:
        p = replace(base_params, seed=seed, manipulation=manip, **overrides)
        ds = generate_dataset(p)
        delta, angle, _ = project_and_measure(ds, p)
        deltas.append(delta)
        angles.append(angle)
    return deltas, angles


def run_block_comparison(
    seeds: list[int] | None = None,
    base_params: MultiblockRecoveryParams | None = None,
    dim_ratios: list[float] | None = None,
) -> pd.DataFrame:
    """Headline comparison: cross-talk vs block count and dim-ratio.

    Sweeps ``n_nuisance_blocks ∈ {0, 1, 2}`` and ``dim_ratios`` for all
    manipulations.  ``n_nuisance_blocks = 0`` rows are the single-block
    Rung-2 baseline (``dim_ratio`` is reported as ``0.0`` for those rows).

    Parameters
    ----------
    seeds:
        RNG seeds to average over.  Defaults to ``list(range(10))``.
    base_params:
        Template configuration.  Defaults to ``MultiblockRecoveryParams()``.
    dim_ratios:
        Nuisance-block dimensionality ratios to sweep.  Defaults to
        ``[0.5, 1.0, 2.0, 5.0, 10.0]``.

    Returns
    -------
    pd.DataFrame
        One row per ``(n_nuisance_blocks, dim_ratio, manipulation)`` with
        columns ``delta_mean``, ``delta_std``, ``angle_mean``, ``angle_std``.
    """
    if seeds is None:
        seeds = list(range(10))
    if base_params is None:
        base_params = MultiblockRecoveryParams()
    if dim_ratios is None:
        dim_ratios = _DEFAULT_DIM_RATIOS

    rows = []

    # Single-block baseline (n_nuisance_blocks = 0)
    for manip in _MANIPULATIONS:
        deltas, angles = _measure_over_seeds(
            replace(base_params, n_nuisance_blocks=0), manip, seeds
        )
        rows.append(
            {
                "n_nuisance_blocks": 0,
                "dim_ratio": 0.0,
                "manipulation": manip,
                "delta_mean": float(np.mean(deltas)),
                "delta_std": float(np.std(deltas)),
                "angle_mean": float(np.mean(angles)),
                "angle_std": float(np.std(angles)),
            }
        )

    # Multi-block arms
    for n_blocks in (1, 2):
        for dr in dim_ratios:
            for manip in _MANIPULATIONS:
                deltas, angles = _measure_over_seeds(
                    base_params, manip, seeds,
                    n_nuisance_blocks=n_blocks, dim_ratio=dr,
                )
                rows.append(
                    {
                        "n_nuisance_blocks": n_blocks,
                        "dim_ratio": dr,
                        "manipulation": manip,
                        "delta_mean": float(np.mean(deltas)),
                        "delta_std": float(np.std(deltas)),
                        "angle_mean": float(np.mean(angles)),
                        "angle_std": float(np.std(angles)),
                    }
                )

    return pd.DataFrame(rows)


def run_dim_ratio_sweep(
    dim_ratios: list[float] | None = None,
    seeds: list[int] | None = None,
    base_params: MultiblockRecoveryParams | None = None,
) -> pd.DataFrame:
    """Primary sweep: dim_ratio vs manipulations (n_nuisance_blocks = 1).

    Includes a ``dim_ratio = 0.0`` baseline row generated with
    ``n_nuisance_blocks = 0`` so the single-block floor appears on the same
    axis.

    Parameters
    ----------
    dim_ratios:
        Values of ``dim_ratio`` to sweep.  Defaults to
        ``[0.5, 1.0, 2.0, 5.0, 10.0]``.
    seeds:
        Defaults to ``list(range(10))``.
    base_params:
        Template configuration (``n_nuisance_blocks`` is overridden).

    Returns
    -------
    pd.DataFrame
        One row per ``(dim_ratio, manipulation)`` with ``delta_mean``,
        ``delta_std``, ``angle_mean``, ``angle_std``.
    """
    if dim_ratios is None:
        dim_ratios = _DEFAULT_DIM_RATIOS
    if seeds is None:
        seeds = list(range(10))
    if base_params is None:
        base_params = MultiblockRecoveryParams()

    rows = []

    # Single-block baseline
    for manip in _MANIPULATIONS:
        deltas, angles = _measure_over_seeds(
            replace(base_params, n_nuisance_blocks=0), manip, seeds
        )
        rows.append(
            {
                "dim_ratio": 0.0,
                "manipulation": manip,
                "delta_mean": float(np.mean(deltas)),
                "delta_std": float(np.std(deltas)),
                "angle_mean": float(np.mean(angles)),
                "angle_std": float(np.std(angles)),
            }
        )

    for dr in dim_ratios:
        for manip in _MANIPULATIONS:
            deltas, angles = _measure_over_seeds(
                base_params, manip, seeds, n_nuisance_blocks=1, dim_ratio=dr
            )
            rows.append(
                {
                    "dim_ratio": dr,
                    "manipulation": manip,
                    "delta_mean": float(np.mean(deltas)),
                    "delta_std": float(np.std(deltas)),
                    "angle_mean": float(np.mean(angles)),
                    "angle_std": float(np.std(angles)),
                }
            )

    return pd.DataFrame(rows)


def run_rho_sweep(
    rho_values: list[float] | None = None,
    dim_ratio: float = 5.0,
    seeds: list[int] | None = None,
    base_params: MultiblockRecoveryParams | None = None,
) -> pd.DataFrame:
    """Secondary sweep: nuisance-block correlation at a fixed high dim_ratio.

    Parameters
    ----------
    rho_values:
        Exchangeable correlation values to sweep.  Defaults to
        ``[0.0, 0.3, 0.7]``.
    dim_ratio:
        Fixed dimensionality ratio.  Defaults to ``5.0`` (the point where
        block-weight imbalance is most pronounced at default settings).
    seeds:
        Defaults to ``list(range(10))``.
    base_params:
        Template configuration.

    Returns
    -------
    pd.DataFrame
        One row per ``(rho_nuisance, manipulation)`` with delta/angle stats.
    """
    if rho_values is None:
        rho_values = _DEFAULT_RHO_VALUES
    if seeds is None:
        seeds = list(range(10))
    if base_params is None:
        base_params = MultiblockRecoveryParams()

    rows = []
    for rho in rho_values:
        for manip in _MANIPULATIONS:
            deltas, angles = _measure_over_seeds(
                base_params, manip, seeds,
                n_nuisance_blocks=1, dim_ratio=dim_ratio, rho_nuisance=rho,
            )
            rows.append(
                {
                    "rho_nuisance": rho,
                    "dim_ratio": dim_ratio,
                    "manipulation": manip,
                    "delta_mean": float(np.mean(deltas)),
                    "delta_std": float(np.std(deltas)),
                    "angle_mean": float(np.mean(angles)),
                    "angle_std": float(np.std(angles)),
                }
            )
    return pd.DataFrame(rows)


def run_effect_size_sweep(
    signal_scales: list[float],
    dim_ratios: list[float] | None = None,
    seeds: list[int] | None = None,
    base_params: MultiblockRecoveryParams | None = None,
) -> pd.DataFrame:
    """Sweep effect size × dim_ratio for all manipulations (n_nuisance_blocks=1).

    Locates whether cross-talk onset depends on effect size.

    Parameters
    ----------
    signal_scales:
        Values of ``signal_scale`` (‖a_feat‖) to sweep.
    dim_ratios:
        Values of ``dim_ratio`` to sweep alongside effect size.  Defaults to
        ``[0.0, 1.0, 5.0, 10.0]`` (``0.0`` = single-block baseline).
    seeds:
        Defaults to ``list(range(10))``.
    base_params:
        Template configuration.

    Returns
    -------
    pd.DataFrame
        One row per ``(signal_scale, dim_ratio, manipulation)`` with
        delta/angle stats.
    """
    if dim_ratios is None:
        dim_ratios = [0.0, 1.0, 5.0, 10.0]
    if seeds is None:
        seeds = list(range(10))
    if base_params is None:
        base_params = MultiblockRecoveryParams()

    rows = []
    for scale in signal_scales:
        for dr in dim_ratios:
            n_blocks = 0 if dr == 0.0 else 1
            for manip in _MANIPULATIONS:
                deltas, angles = _measure_over_seeds(
                    base_params, manip, seeds,
                    signal_scale=scale,
                    n_nuisance_blocks=n_blocks,
                    dim_ratio=dr if n_blocks > 0 else base_params.dim_ratio,
                )
                rows.append(
                    {
                        "signal_scale": float(scale),
                        "dim_ratio": dr,
                        "manipulation": manip,
                        "delta_mean": float(np.mean(deltas)),
                        "delta_std": float(np.std(deltas)),
                        "angle_mean": float(np.mean(angles)),
                        "angle_std": float(np.std(angles)),
                    }
                )
    return pd.DataFrame(rows)


def run_block_weight_curve(
    dim_ratios: list[float] | None = None,
    n_nuisance_blocks: int = 1,
    seeds: list[int] | None = None,
    base_params: MultiblockRecoveryParams | None = None,
) -> pd.DataFrame:
    """Anchor block-weight fraction vs dim_ratio.

    Quantifies how much of the top-k PCA loading mass falls in the anchor
    block as nuisance dimensionality grows.  Cross-talk onset is expected to
    coincide with ``w_anchor`` dropping below 0.5.

    Parameters
    ----------
    dim_ratios:
        Defaults to ``[0.5, 1.0, 2.0, 5.0, 10.0]``.
    n_nuisance_blocks:
        Number of nuisance blocks.  Defaults to ``1``.
    seeds:
        Seeds to average ``w_anchor`` over.  Defaults to ``[0, 1, 2]`` (the
        block weight is structurally stable across seeds; a small set suffices).
    base_params:
        Template configuration.

    Returns
    -------
    pd.DataFrame
        One row per ``(dim_ratio, seed)`` with columns ``w_anchor``,
        ``n_features_anchor``, ``n_features_nuisance``, ``n_features_total``,
        ``p_anchor_naive`` (``p_anchor / p_total``, the expected weight under
        uncorrelated equal-scale blocks).
    """
    if dim_ratios is None:
        dim_ratios = _DEFAULT_DIM_RATIOS
    if seeds is None:
        seeds = [0, 1, 2]
    if base_params is None:
        base_params = MultiblockRecoveryParams()

    rows = []

    # Single-block reference (w_anchor = 1.0 by construction).
    for seed in seeds:
        rows.append(
            {
                "dim_ratio": 0.0,
                "seed": seed,
                "w_anchor": 1.0,
                "n_features_anchor": base_params.n_features_anchor,
                "n_features_nuisance": 0,
                "n_features_total": base_params.n_features_anchor,
                "p_anchor_naive": 1.0,
            }
        )

    for dr in dim_ratios:
        p = replace(
            base_params,
            n_nuisance_blocks=n_nuisance_blocks,
            dim_ratio=dr,
            manipulation="none",
        )
        p_a = p.n_features_anchor
        p_nuis = p.n_features_nuisance
        p_total = p.n_features_total
        p_naive = p_a / p_total

        for seed in seeds:
            ps = replace(p, seed=seed)
            ds = generate_dataset(ps)
            X_joint, _ = build_joint_matrix(ds)
            w_anchor, _ = decompose_block_weight(X_joint, p_a, ps.n_components)
            rows.append(
                {
                    "dim_ratio": dr,
                    "seed": seed,
                    "w_anchor": w_anchor,
                    "n_features_anchor": p_a,
                    "n_features_nuisance": p_nuis,
                    "n_features_total": p_total,
                    "p_anchor_naive": float(p_naive),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_dim_ratio_sweep(sweep: pd.DataFrame) -> Figure:
    """Line plot of delta/angle vs dim_ratio per manipulation.

    Two panels (delta, angle); x-axis is ``dim_ratio`` (0 = single-block
    baseline); one line per manipulation with shaded ±1 SD band.

    Parameters
    ----------
    sweep:
        Output of :func:`run_dim_ratio_sweep`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    colors = {"none": "0.5", "magnitude": "C0", "orientation": "C1"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for stat, ax in zip(("delta", "angle"), axes):
        for manip in _MANIPULATIONS:
            sub = sweep[sweep["manipulation"] == manip].sort_values("dim_ratio")
            x = sub["dim_ratio"].to_numpy()
            y = sub[f"{stat}_mean"].to_numpy()
            ye = sub[f"{stat}_std"].to_numpy()
            ax.plot(x, y, marker="o", color=colors[manip], label=manip)
            ax.fill_between(x, y - ye, y + ye, alpha=0.2, color=colors[manip])
        ax.set_xlabel("dim_ratio  (p_nuisance / p_anchor;  0 = single-block)")
        ax.set_ylabel(f"measured {stat}" + ("  (degrees)" if stat == "angle" else ""))
        ax.set_title(f"{stat} vs nuisance-block dimensionality ratio")
        ax.legend(title="manipulation", fontsize=8)

    fig.suptitle(
        "Rung 3 — Multi-block concatenation: delta/angle vs dim_ratio",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    return fig


def plot_block_weight_curve(weight_df: pd.DataFrame) -> Figure:
    """Line plot of anchor block-weight fraction vs dim_ratio.

    Shows ``w_anchor`` (mean ± SD over seeds) and the naive feature-fraction
    ``p_anchor / p_total`` as a dashed reference.  If cross-talk onset
    coincides with ``w_anchor`` dropping below 0.5, the two curves overlap at
    that point.

    Parameters
    ----------
    weight_df:
        Output of :func:`run_block_weight_curve`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    summary = (
        weight_df.groupby("dim_ratio")
        .agg(
            w_mean=("w_anchor", "mean"),
            w_std=("w_anchor", "std"),
            p_naive=("p_anchor_naive", "mean"),
        )
        .reset_index()
        .sort_values("dim_ratio")
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    x = summary["dim_ratio"].to_numpy()
    ax.plot(x, summary["w_mean"].to_numpy(), marker="o", color="C2", label="w_anchor")
    ax.fill_between(
        x,
        summary["w_mean"].to_numpy() - summary["w_std"].to_numpy(),
        summary["w_mean"].to_numpy() + summary["w_std"].to_numpy(),
        alpha=0.2,
        color="C2",
    )
    ax.plot(
        x,
        summary["p_naive"].to_numpy(),
        linestyle="--",
        color="0.5",
        label="p_anchor / p_total  (naive)",
    )
    ax.axhline(0.5, linestyle=":", color="0.3", linewidth=0.8)
    ax.set_xlabel("dim_ratio  (p_nuisance / p_anchor;  0 = single-block)")
    ax.set_ylabel("anchor block-weight fraction  w_anchor")
    ax.set_ylim(0, 1.05)
    ax.set_title("Rung 3 — Anchor loading fraction in top-k PCA components")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig
