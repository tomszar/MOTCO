"""Projector test bed for trajectory recovery (Rung 2).

Isolates the **projector** (the dimensionality-reduction / integration step) as
the single factor on top of the clean linear floor. The known 2-stage geometry
(``none``/``magnitude``/``orientation``) is injected exactly as in Rung 0 into a
clean linear feature space — the methylation **M-value frame**, so there is *no*
``rev.logit`` distortion (Rung 1 established M-value integration as the correct
representation). The only thing that varies versus the Rung-0 clean floor is the
projector used before measurement:

- ``pca``         — mean-centered PCA (the Rung-0/1 reference floor).
- ``standardize`` — per-feature z-score (the production ``concat`` transform),
                    then PCA.
- ``plsda``       — supervised PLS-DA latent space, conditioned on the group
                    label (``stats/pls.fit_plsda_transform``).
- ``snf``         — graph-spectral embedding (``stats/snf``). SNF *fusion*
                    requires ≥ 2 networks, so on this single-block test bed the
                    arm reduces to ``get_affinity_matrix`` → ``get_spectral`` —
                    the per-block core of the production ``snf`` path. This is a
                    faithful single-block analogue and is documented as such.

Measurement (``get_model_matrix``/``build_ls_means``/``estimate_difference`` on a
2-group × 2-stage design, two-group contrast) is identical to Rung 0/1. Cross-talk
is read against each projector's own ``none`` null and against the PCA floor.

This is Rung 2 of a ladder. Rung 0 (``linear_recovery``) proved the linear floor
is clean; Rung 1 (``methylation_recovery``) showed ``rev.logit`` is inverted by
M-value integration; Rung 2 asks whether the projector itself manufactures
magnitude→orientation cross-talk.

References
----------
openspec/changes/rung2-projector-crosstalk/design.md
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.decomposition import PCA

from motco.simulations.linear_recovery import (
    LinearRecoveryParams,
)
from motco.simulations.linear_recovery import (
    generate_dataset as _linear_generate_dataset,
)
from motco.stats.design import build_ls_means, get_model_matrix
from motco.stats.pls import fit_plsda_transform
from motco.stats.snf import get_affinity_matrix, get_spectral
from motco.stats.trajectory import estimate_difference

__all__ = [
    "ProjectorRecoveryError",
    "Projector",
    "ProjectorRecoveryParams",
    "ProjectorRecoveryDataset",
    "generate_dataset",
    "project",
    "project_and_measure",
    "run_projector_comparison",
    "run_dimensionality_sweep",
    "run_effect_size_sweep",
    "run_leakage_probe",
    "plot_projector_comparison",
]


class ProjectorRecoveryError(ValueError):
    """Raised on invalid ``ProjectorRecoveryParams``."""


Projector = Literal["pca", "standardize", "plsda", "snf"]
_PROJECTORS: list[Projector] = ["pca", "standardize", "plsda", "snf"]
_MANIPULATIONS: list[Literal["none", "magnitude", "orientation"]] = [
    "none",
    "magnitude",
    "orientation",
]


@dataclass(frozen=True)
class ProjectorRecoveryParams:
    """Frozen configuration for a Rung-2 projector recovery run.

    Mirrors :class:`~motco.simulations.linear_recovery.LinearRecoveryParams`
    (the geometry is injected identically) and adds the projector selector plus
    an optional anisotropy knob and SNF affinity parameters.

    Parameters
    ----------
    seed:
        RNG seed for deterministic generation.
    n_features:
        Dimensionality of the (M-value) feature space.
    n_samples_per_cell:
        Samples drawn per (group, stage) cell.
    noise_scale:
        Base standard deviation of the Gaussian noise.
    signal_scale:
        ‖a_feat‖ — the magnitude of group A's feature-space step.
    manipulation:
        Geometric transform applied to group B's step vector.
    scale_c:
        Magnitude scale for the ``"magnitude"`` manipulation (step_B = c · a_feat).
    angle_theta:
        Rotation angle in degrees for the ``"orientation"`` manipulation.
    n_components:
        Number of latent components retained for measurement. Defaults to ``2``:
        the 2-group × 2-stage signal lives in ≤ 2 dimensions, and Rung 0 showed
        the angle null floor grows with each retained *noise* PC. The headline
        run keeps only the signal subspace so the *projector* — not the k-noise
        floor — is the variable under test; the dimensionality sweep probes higher.
    projector:
        Which projector to apply before measurement (``pca``/``standardize``/
        ``plsda``/``snf``).
    anisotropy:
        Heteroscedasticity of the per-feature noise. ``0.0`` (default) is
        isotropic (every feature shares ``noise_scale``); ``> 0`` draws
        deterministic per-feature std multipliers ``exp(anisotropy · z_i)`` so
        per-feature scales differ — making the ``standardize`` arm non-trivial.
    snf_K:
        Nearest-neighbor count for the SNF affinity matrix.
    snf_eps:
        Normalization factor for the SNF affinity matrix.
    """

    seed: int = 0
    n_features: int = 50
    n_samples_per_cell: int = 40
    noise_scale: float = 1.0
    signal_scale: float = 5.0
    manipulation: Literal["none", "magnitude", "orientation"] = "none"
    scale_c: float = 2.0
    angle_theta: float = 45.0
    n_components: int = 2
    projector: Projector = "pca"
    anisotropy: float = 0.0
    snf_K: int = 20
    snf_eps: float = 0.5

    def as_linear(self) -> LinearRecoveryParams:
        """Project to the Rung-0 params (shared fields) for step-construction reuse."""
        return LinearRecoveryParams(
            seed=self.seed,
            n_features=self.n_features,
            n_samples_per_cell=self.n_samples_per_cell,
            noise_scale=self.noise_scale,
            signal_scale=self.signal_scale,
            manipulation=self.manipulation,
            scale_c=self.scale_c,
            angle_theta=self.angle_theta,
            n_components=self.n_components,
        )


@dataclass
class ProjectorRecoveryDataset:
    """Output of :func:`generate_dataset`.

    Attributes
    ----------
    X:
        Feature matrix (n_samples × n_features) in the clean M-value frame.
    metadata:
        Sample metadata with ``"group"`` (A/B) and ``"stage"`` (0/1) columns,
        row-aligned with ``X``.
    step_A:
        Ground-truth feature-space step for group A (μ_{A,1} − μ_{A,0}).
    step_B:
        Ground-truth feature-space step for group B.
    """

    X: pd.DataFrame
    metadata: pd.DataFrame
    step_A: np.ndarray
    step_B: np.ndarray


# ---------------------------------------------------------------------------
# Validation and generation
# ---------------------------------------------------------------------------


def _validate(p: ProjectorRecoveryParams) -> None:
    if p.n_samples_per_cell < 2:
        raise ProjectorRecoveryError("n_samples_per_cell must be >= 2")
    if p.n_components < 2:
        raise ProjectorRecoveryError("n_components must be >= 2")
    if p.n_components > p.n_features:
        raise ProjectorRecoveryError(
            f"n_components ({p.n_components}) must be <= n_features ({p.n_features})"
        )
    n_samples = 4 * p.n_samples_per_cell
    if p.n_components >= n_samples:
        raise ProjectorRecoveryError(
            f"n_components ({p.n_components}) must be < n_samples ({n_samples})"
        )
    if p.noise_scale <= 0:
        raise ProjectorRecoveryError("noise_scale must be > 0")
    if p.signal_scale <= 0:
        raise ProjectorRecoveryError("signal_scale must be > 0")
    if p.anisotropy < 0:
        raise ProjectorRecoveryError("anisotropy must be >= 0")
    if p.manipulation not in ("none", "magnitude", "orientation"):
        raise ProjectorRecoveryError(
            f"manipulation must be 'none', 'magnitude', or 'orientation'; "
            f"got {p.manipulation!r}"
        )
    if p.projector not in _PROJECTORS:
        raise ProjectorRecoveryError(
            f"projector must be one of {_PROJECTORS}; got {p.projector!r}"
        )
    if p.projector == "snf":
        if p.snf_K < 1 or p.snf_K >= n_samples:
            raise ProjectorRecoveryError(
                f"snf_K ({p.snf_K}) must be in [1, n_samples-1] (n_samples={n_samples})"
            )
        if p.snf_eps <= 0:
            raise ProjectorRecoveryError("snf_eps must be > 0")


def generate_dataset(params: ProjectorRecoveryParams) -> ProjectorRecoveryDataset:
    """Generate a clean linear test-bed dataset with known trajectory geometry.

    The per-group steps (``step_A``, ``step_B``) are constructed by the Rung-0
    generator (reused unchanged), so the injected geometry is identical to Rung 0.
    Means are ``μ_{g,0} = 0`` and ``μ_{g,1} = step_g``; samples are drawn
    ``x = μ + N(0, Σ)`` where ``Σ`` is diagonal with per-feature std
    ``noise_scale · exp(anisotropy · z_i)`` (isotropic when ``anisotropy = 0``).

    Parameters
    ----------
    params:
        Generation configuration.
    """
    _validate(params)

    # Reuse Rung-0 step construction (deterministic given seed); ignore its X.
    lin = _linear_generate_dataset(params.as_linear())
    step_A, step_B = lin.step_A, lin.step_B

    p = params.n_features
    n = params.n_samples_per_cell
    mu: dict[tuple[str, str], np.ndarray] = {
        ("A", "0"): np.zeros(p),
        ("A", "1"): step_A,
        ("B", "0"): np.zeros(p),
        ("B", "1"): step_B,
    }

    rng = np.random.default_rng(params.seed)
    if params.anisotropy > 0:
        feat_scale = np.exp(params.anisotropy * rng.standard_normal(p))
    else:
        feat_scale = np.ones(p)
    noise_std = params.noise_scale * feat_scale  # per-feature std

    rows: list[np.ndarray] = []
    meta_rows: list[tuple[str, str]] = []
    for group in ("A", "B"):
        for stage in ("0", "1"):
            noise = rng.standard_normal((n, p)) * noise_std
            rows.append(mu[(group, stage)] + noise)
            meta_rows.extend([(group, stage)] * n)

    X_arr = np.vstack(rows)
    X_df = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(p)])
    metadata = pd.DataFrame(meta_rows, columns=["group", "stage"])

    return ProjectorRecoveryDataset(
        X=X_df, metadata=metadata, step_A=step_A, step_B=step_B
    )


# ---------------------------------------------------------------------------
# Projectors (uniform (X, y) → Y contract)
# ---------------------------------------------------------------------------


def project(
    dataset: ProjectorRecoveryDataset, params: ProjectorRecoveryParams
) -> pd.DataFrame:
    """Map the feature matrix to an ``n_components`` latent matrix.

    All four projectors share the same contract: take the
    ``(n_samples × n_features)`` feature matrix (and, for the supervised arm, the
    group label) and return an ``(n_samples × n_components)`` latent ``DataFrame``
    whose rows are aligned with ``dataset.metadata``.

    - ``pca``         — ``PCA`` on mean-centered features (sklearn centers
                        internally); reproduces the Rung-0 floor exactly.
    - ``standardize`` — per-feature z-score (``std == 0 → 1``, matching
                        ``evaluation.py:_concat_integration``) then ``PCA``.
    - ``plsda``       — ``fit_plsda_transform`` with the group label.
    - ``snf``         — ``get_affinity_matrix`` → ``get_spectral`` (single-block;
                        SNF fusion needs ≥ 2 networks and is a no-op for one block).
    """
    X = dataset.X.to_numpy(dtype=float)
    k = params.n_components

    if params.projector == "pca":
        Y = PCA(n_components=k).fit_transform(X)
    elif params.projector == "standardize":
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True)
        std[std == 0.0] = 1.0
        Xz = (X - mean) / std
        Y = PCA(n_components=k).fit_transform(Xz)
    elif params.projector == "plsda":
        y = dataset.metadata["group"].astype(str).to_numpy()
        Y = fit_plsda_transform(X, y, n_components=k)
    else:  # snf — single-block: affinity → spectral (fusion mooted for one block)
        aff = get_affinity_matrix([X], K=params.snf_K, eps=params.snf_eps)[0]
        Y = get_spectral(aff, n_components=k)

    return pd.DataFrame(
        np.asarray(Y, dtype=float),
        columns=[f"comp{i + 1}" for i in range(k)],
        index=dataset.metadata.index,
    )


def _measure_projected(
    Y: pd.DataFrame, metadata: pd.DataFrame
) -> tuple[float, float]:
    """Measure group-A-vs-B delta/angle on a projected outcome matrix.

    Builds the 2-group × 2-stage design and two-group contrast and calls
    ``estimate_difference`` — identical to the Rung-0/1 measurement path.
    """
    model_matrix = get_model_matrix(
        metadata, group_col="group", level_col="stage", full=True
    )
    g_levels = sorted(metadata["group"].astype(str).unique().tolist())
    l_levels = sorted(metadata["stage"].astype(str).unique().tolist())
    ls_means = build_ls_means(g_levels, l_levels, full=True)
    n_l = len(l_levels)
    contrast = [list(range(i * n_l, (i + 1) * n_l)) for i in range(len(g_levels))]
    deltas, angles, _ = estimate_difference(Y, model_matrix, ls_means, contrast)
    return float(deltas[0, 1]), float(angles[0, 1])


def project_and_measure(
    dataset: ProjectorRecoveryDataset, params: ProjectorRecoveryParams
) -> tuple[float, float, pd.DataFrame]:
    """Project with the selected projector and measure delta/angle.

    Returns
    -------
    delta : float
        Group-A vs Group-B magnitude difference (projector-relative for ``snf``).
    angle : float
        Group-A vs Group-B direction difference (degrees).
    Y : pd.DataFrame
        Projected outcome matrix (n_samples × n_components).
    """
    Y = project(dataset, params)
    delta, angle = _measure_projected(Y, dataset.metadata)
    return delta, angle, Y


# ---------------------------------------------------------------------------
# Drivers / sweeps
# ---------------------------------------------------------------------------


def _measure_over_seeds(
    base_params: ProjectorRecoveryParams,
    manip: Literal["none", "magnitude", "orientation"],
    seeds: list[int],
    **overrides: Any,
) -> tuple[list[float], list[float]]:
    deltas: list[float] = []
    angles: list[float] = []
    for seed in seeds:
        params = replace(base_params, seed=seed, manipulation=manip, **overrides)
        dataset = generate_dataset(params)
        delta, angle, _ = project_and_measure(dataset, params)
        deltas.append(delta)
        angles.append(angle)
    return deltas, angles


def run_projector_comparison(
    seeds: list[int] | None = None,
    base_params: ProjectorRecoveryParams | None = None,
    projectors: list[Projector] | None = None,
) -> pd.DataFrame:
    """Compare projectors on identical geometry over the seed set.

    Returns a tidy DataFrame with one row per (projector, manipulation):
    ``projector``, ``manipulation``, and the mean ± SD of ``delta``/``angle``
    over ``seeds``. The ``none`` rows give each projector's null floor; the
    ``pca`` rows are the cross-projector reference.

    Parameters
    ----------
    seeds:
        RNG seeds to average over. Defaults to ``list(range(10))``.
    base_params:
        Template configuration. Defaults to ``ProjectorRecoveryParams()``.
    projectors:
        Projectors to compare. Defaults to all four.
    """
    if seeds is None:
        seeds = list(range(10))
    if base_params is None:
        base_params = ProjectorRecoveryParams()
    if projectors is None:
        projectors = list(_PROJECTORS)

    rows = []
    for projector in projectors:
        params = replace(base_params, projector=projector)
        for manip in _MANIPULATIONS:
            deltas, angles = _measure_over_seeds(params, manip, seeds)
            rows.append(
                {
                    "projector": projector,
                    "manipulation": manip,
                    "delta_mean": float(np.mean(deltas)),
                    "delta_std": float(np.std(deltas)),
                    "angle_mean": float(np.mean(angles)),
                    "angle_std": float(np.std(angles)),
                }
            )
    return pd.DataFrame(rows)


def run_dimensionality_sweep(
    n_components_grid: list[int],
    seeds: list[int] | None = None,
    base_params: ProjectorRecoveryParams | None = None,
    projectors: list[Projector] | None = None,
) -> pd.DataFrame:
    """Sweep retained latent dimensionality per projector and manipulation.

    Locates whether any projector-induced cross-talk is a dimensionality
    artifact (too few/many retained directions) or intrinsic. Same column
    layout as :func:`run_projector_comparison` plus an ``n_components`` column.
    """
    if seeds is None:
        seeds = list(range(10))
    if base_params is None:
        base_params = ProjectorRecoveryParams()
    if projectors is None:
        projectors = list(_PROJECTORS)

    rows = []
    for k in n_components_grid:
        for projector in projectors:
            params = replace(base_params, projector=projector, n_components=k)
            for manip in _MANIPULATIONS:
                deltas, angles = _measure_over_seeds(params, manip, seeds)
                rows.append(
                    {
                        "n_components": int(k),
                        "projector": projector,
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
    seeds: list[int] | None = None,
    base_params: ProjectorRecoveryParams | None = None,
    projectors: list[Projector] | None = None,
) -> pd.DataFrame:
    """Sweep effect size (``signal_scale``) per projector and manipulation.

    Same column layout as :func:`run_projector_comparison` plus a
    ``signal_scale`` column. Locates the onset of any projector-induced
    distortion as the injected step grows.
    """
    if seeds is None:
        seeds = list(range(10))
    if base_params is None:
        base_params = ProjectorRecoveryParams()
    if projectors is None:
        projectors = list(_PROJECTORS)

    rows = []
    for scale in signal_scales:
        for projector in projectors:
            params = replace(base_params, projector=projector, signal_scale=scale)
            for manip in _MANIPULATIONS:
                deltas, angles = _measure_over_seeds(params, manip, seeds)
                rows.append(
                    {
                        "signal_scale": float(scale),
                        "projector": projector,
                        "manipulation": manip,
                        "delta_mean": float(np.mean(deltas)),
                        "delta_std": float(np.std(deltas)),
                        "angle_mean": float(np.mean(angles)),
                        "angle_std": float(np.std(angles)),
                    }
                )
    return pd.DataFrame(rows)


def run_leakage_probe(
    n_components_grid: list[int],
    seeds: list[int] | None = None,
    base_params: ProjectorRecoveryParams | None = None,
) -> pd.DataFrame:
    """Supervised-leakage probe: ``none`` trajectory under PLS-DA vs PCA.

    PLS-DA conditions the projection on the group label, so it can manufacture
    group-aligned structure even when the two groups have *identical* (``none``)
    trajectories. This probe measures the ``none`` ``delta``/``angle`` under both
    ``plsda`` and ``pca`` across a component grid (larger ``n_components`` relative
    to the sample size stresses the leakage), so any inflation of the PLS-DA null
    above the PCA floor is exposed.

    Returns one row per (n_components, projector ∈ {pca, plsda}) with the ``none``
    ``delta``/``angle`` mean ± SD.
    """
    if seeds is None:
        seeds = list(range(10))
    if base_params is None:
        base_params = ProjectorRecoveryParams()

    rows = []
    for k in n_components_grid:
        for projector in ("pca", "plsda"):
            params = replace(base_params, projector=projector, n_components=k)
            deltas, angles = _measure_over_seeds(params, "none", seeds)
            rows.append(
                {
                    "n_components": int(k),
                    "projector": projector,
                    "delta_mean": float(np.mean(deltas)),
                    "delta_std": float(np.std(deltas)),
                    "angle_mean": float(np.mean(angles)),
                    "angle_std": float(np.std(angles)),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_projector_comparison(comparison: pd.DataFrame) -> Figure:
    """Grouped bar chart of delta/angle per projector for each manipulation.

    Two panels (delta, angle); x-axis grouped by projector, bars per
    manipulation, with the ``none`` null floor visible per projector.

    Parameters
    ----------
    comparison:
        Output of :func:`run_projector_comparison`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    projectors = list(dict.fromkeys(comparison["projector"]))
    x = np.arange(len(projectors))
    width = 0.25
    colors = {"none": "0.6", "magnitude": "C0", "orientation": "C1"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for stat, ax in zip(("delta", "angle"), axes):
        for j, manip in enumerate(_MANIPULATIONS):
            sub = comparison[comparison["manipulation"] == manip].set_index("projector")
            means = [sub.loc[p, f"{stat}_mean"] for p in projectors]
            errs = [sub.loc[p, f"{stat}_std"] for p in projectors]
            ax.bar(
                x + (j - 1) * width,
                means,
                width,
                yerr=errs,
                capsize=3,
                color=colors[manip],
                label=manip,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(projectors)
        ax.set_ylabel(f"measured {stat}" + ("  (degrees)" if stat == "angle" else ""))
        ax.set_title(f"{stat} by projector")
        ax.legend(title="manipulation", fontsize=8)

    fig.suptitle(
        "Rung 2 — Projector recovery: delta/angle by projector",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    return fig
