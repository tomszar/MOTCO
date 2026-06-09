"""Gaussian existence-proof test bed for linear trajectory recovery (Rung 0).

Injects a known 2-stage feature-space geometry for two groups under isotropic
MVN noise, projects through inline PCA (mean-centered, no standardization), and
measures delta/angle via the production estimators in ``stats/trajectory.py``.

This is the bottom rung of a ladder that will add the methylation ``rev.logit``
nonlinearity (Rung 1) and the full InterSIM generator (Rung 2). Rung 0 isolates
the linear-algebra floor: if recovery leaks here, the estimator/projector is
implicated, not biology.

References
----------
openspec/changes/rung0-gaussian-existence-proof/design.md
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.decomposition import PCA

from motco.stats.design import build_ls_means, get_model_matrix
from motco.stats.trajectory import estimate_difference


class LinearRecoveryError(ValueError):
    """Raised on invalid ``LinearRecoveryParams``."""


@dataclass(frozen=True)
class LinearRecoveryParams:
    """Frozen configuration for a Gaussian existence-proof run.

    Parameters
    ----------
    seed:
        RNG seed for deterministic generation.
    n_features:
        Dimensionality of feature space.
    n_samples_per_cell:
        Samples drawn per (group, stage) cell.
    noise_scale:
        Standard deviation of isotropic Gaussian noise (Σ = noise_scale² I).
    signal_scale:
        ‖a_feat‖ — the magnitude of group A's feature-space step.
    manipulation:
        Geometric transform applied to group B's step vector.
    scale_c:
        Magnitude scale for the ``"magnitude"`` manipulation (step_B = c · a_feat).
    angle_theta:
        Rotation angle in degrees for the ``"orientation"`` manipulation.
    n_components:
        Number of PCA components retained for measurement.
    """

    seed: int = 0
    n_features: int = 50
    n_samples_per_cell: int = 40
    noise_scale: float = 1.0
    signal_scale: float = 5.0
    manipulation: Literal["none", "magnitude", "orientation"] = "none"
    scale_c: float = 2.0
    angle_theta: float = 45.0
    n_components: int = 10


@dataclass
class LinearRecoveryDataset:
    """Output of ``generate_dataset``.

    Attributes
    ----------
    X:
        Feature matrix (n_samples × n_features).
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


def _validate(p: LinearRecoveryParams) -> None:
    if p.n_samples_per_cell < 2:
        raise LinearRecoveryError("n_samples_per_cell must be >= 2")
    if p.n_components < 2:
        raise LinearRecoveryError("n_components must be >= 2")
    if p.n_components > p.n_features:
        raise LinearRecoveryError(
            f"n_components ({p.n_components}) must be <= n_features ({p.n_features})"
        )
    if p.noise_scale <= 0:
        raise LinearRecoveryError("noise_scale must be > 0")
    if p.signal_scale <= 0:
        raise LinearRecoveryError("signal_scale must be > 0")
    if p.manipulation not in ("none", "magnitude", "orientation"):
        raise LinearRecoveryError(
            f"manipulation must be 'none', 'magnitude', or 'orientation'; "
            f"got {p.manipulation!r}"
        )


def generate_dataset(params: LinearRecoveryParams) -> LinearRecoveryDataset:
    """Generate a Gaussian test-bed dataset with known trajectory geometry.

    Group A's step ``a_feat`` is a random unit vector scaled by
    ``params.signal_scale``.  Group B's step is a deterministic transform of
    ``a_feat`` according to ``params.manipulation``:

    - ``"none"``        → step_B = a_feat  (null control)
    - ``"magnitude"``   → step_B = c · a_feat  (same direction, scaled length)
    - ``"orientation"`` → step_B = ‖a_feat‖ · (cos θ · â + sin θ · û)
                          where û ⊥ â is built via Gram-Schmidt

    Both groups start at the origin at stage 0, so
    μ_{g,1} = step_g and μ_{g,0} = 0.

    Parameters
    ----------
    params:
        Generation configuration.
    """
    _validate(params)
    rng = np.random.default_rng(params.seed)
    p = params.n_features

    a_dir = rng.standard_normal(p)
    a_dir /= np.linalg.norm(a_dir)
    a_feat = params.signal_scale * a_dir

    if params.manipulation == "none":
        step_B = a_feat.copy()
    elif params.manipulation == "magnitude":
        step_B = params.scale_c * a_feat
    else:  # orientation
        theta = np.deg2rad(params.angle_theta)
        v = rng.standard_normal(p)
        v -= np.dot(v, a_dir) * a_dir
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-12:
            idx = int(np.argmin(np.abs(a_dir)))
            v = np.zeros(p)
            v[idx] = 1.0
            v -= np.dot(v, a_dir) * a_dir
            v_norm = np.linalg.norm(v)
        u_dir = v / v_norm
        step_B = params.signal_scale * (np.cos(theta) * a_dir + np.sin(theta) * u_dir)

    mu: dict[tuple[str, str], np.ndarray] = {
        ("A", "0"): np.zeros(p),
        ("A", "1"): a_feat,
        ("B", "0"): np.zeros(p),
        ("B", "1"): step_B,
    }

    n = params.n_samples_per_cell
    rows: list[np.ndarray] = []
    meta_rows: list[tuple[str, str]] = []
    for group in ("A", "B"):
        for stage in ("0", "1"):
            noise = rng.standard_normal((n, p)) * params.noise_scale
            rows.append(mu[(group, stage)] + noise)
            meta_rows.extend([(group, stage)] * n)

    X_arr = np.vstack(rows)
    X_df = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(p)])
    metadata = pd.DataFrame(meta_rows, columns=["group", "stage"])

    return LinearRecoveryDataset(X=X_df, metadata=metadata, step_A=a_feat, step_B=step_B)


# ---------------------------------------------------------------------------
# Projection and measurement
# ---------------------------------------------------------------------------


def project_and_measure(
    dataset: LinearRecoveryDataset,
    params: LinearRecoveryParams,
) -> tuple[float, float, PCA, pd.DataFrame, np.ndarray]:
    """Fit PCA, project, and measure delta/angle via the production estimators.

    PCA is fit on mean-centered features with no per-feature standardization
    (Rung-0 design choice).  The projected outcome matrix is then passed to
    ``estimate_difference`` with a 2-group × 2-stage design constructed via
    ``get_model_matrix`` and ``build_ls_means``.

    Parameters
    ----------
    dataset:
        Output of ``generate_dataset``.
    params:
        Must carry the same ``n_components`` used to build ``dataset``.

    Returns
    -------
    delta : float
        Group-A vs Group-B magnitude difference.
    angle : float
        Group-A vs Group-B direction difference (degrees).
    pca : sklearn.decomposition.PCA
        Fitted projector (exposes ``components_``, ``explained_variance_ratio_``).
    Y : pd.DataFrame
        Projected outcome matrix (n_samples × n_components).
    Vk : np.ndarray
        Loadings, shape (n_features × n_components) = ``pca.components_.T``.
    """
    X_arr = dataset.X.to_numpy(dtype=float)
    pca = PCA(n_components=params.n_components)
    Y_arr = pca.fit_transform(X_arr)
    Y = pd.DataFrame(Y_arr, columns=[f"PC{i + 1}" for i in range(params.n_components)])
    Vk = pca.components_.T  # (n_features, n_components)

    metadata = dataset.metadata
    model_matrix = get_model_matrix(
        metadata, group_col="group", level_col="stage", full=True
    )

    g_levels = sorted(metadata["group"].astype(str).unique().tolist())
    l_levels = sorted(metadata["stage"].astype(str).unique().tolist())
    ls_means = build_ls_means(g_levels, l_levels, full=True)

    # LS-mean row order is group-major, level-minor: [A/0, A/1, B/0, B/1]
    n_l = len(l_levels)
    contrast = [list(range(i * n_l, (i + 1) * n_l)) for i in range(len(g_levels))]

    deltas, angles, _ = estimate_difference(Y, model_matrix, ls_means, contrast)
    return float(deltas[0, 1]), float(angles[0, 1]), pca, Y, Vk


# ---------------------------------------------------------------------------
# Exact inverse design
# ---------------------------------------------------------------------------


def givens_rotation(k: int, theta: float, i: int = 0, j: int = 1) -> np.ndarray:
    """Build a k×k Givens rotation in the (i, j) plane by ``theta`` radians.

    The matrix is orthogonal and length-preserving by construction.

    Parameters
    ----------
    k:
        Dimension of the latent space.
    theta:
        Rotation angle in radians.
    i, j:
        Indices of the 2D rotation plane (default: PC1 / PC2).
    """
    R = np.eye(k)
    c, s = np.cos(theta), np.sin(theta)
    R[i, i] = c
    R[j, j] = c
    R[i, j] = -s
    R[j, i] = s
    return R


def inverse_design_magnitude(
    a: np.ndarray, Vk: np.ndarray, c: float
) -> np.ndarray:
    """Minimum-norm feature change that achieves a magnitude-c latent step.

    Given group A's latent step ``a``, the target for group B is ``c · a``.
    The difference is ``(c − 1) · a``, and the minimum-norm feature preimage is
    ``Δx = Vk @ (c − 1) · a``.

    Because ``Vk.T @ Vk = I_k`` (PCA components are orthonormal), the
    round-trip ``Vk.T @ Δx = (c − 1) · a`` holds exactly up to floating-point
    precision.

    Parameters
    ----------
    a:
        Latent-space step vector for group A, shape ``(k,)``.
    Vk:
        PCA loadings (n_features × n_components = ``pca.components_.T``).
    c:
        Magnitude scale.

    Returns
    -------
    np.ndarray
        Feature-space change vector ``Δx``, shape ``(n_features,)``.

    Raises
    ------
    LinearRecoveryError
        If the round-trip assertion fails (indicates a non-orthonormal Vk).
    """
    delta_latent = (c - 1.0) * a
    delta_x = Vk @ delta_latent
    if not np.allclose(Vk.T @ delta_x, delta_latent, atol=1e-10):
        raise LinearRecoveryError("Round-trip failed: Vk.T @ Δx ≠ (c−1)·a")
    return delta_x


def inverse_design_orientation(
    a: np.ndarray, Vk: np.ndarray, R: np.ndarray
) -> np.ndarray:
    """Minimum-norm feature change that achieves an R-rotated latent step.

    The target step for group B in latent space is ``R @ a``.  The difference
    is ``(R − I) @ a``, and the minimum-norm feature preimage is
    ``Δx = Vk @ (R − I) @ a``.

    Parameters
    ----------
    a:
        Latent-space step vector for group A, shape ``(k,)``.
    Vk:
        PCA loadings (n_features × n_components).
    R:
        k×k rotation matrix (e.g. from ``givens_rotation``).

    Returns
    -------
    np.ndarray
        Feature-space change vector ``Δx``, shape ``(n_features,)``.

    Raises
    ------
    LinearRecoveryError
        If the round-trip assertion fails.
    """
    k = len(a)
    delta_latent = (R - np.eye(k)) @ a
    delta_x = Vk @ delta_latent
    if not np.allclose(Vk.T @ delta_x, delta_latent, atol=1e-10):
        raise LinearRecoveryError("Round-trip failed: Vk.T @ Δx ≠ (R−I)·a")
    return delta_x


def delta_x_summary(
    delta_x: np.ndarray, threshold: float = 0.05
) -> dict[str, object]:
    """Support, sparsity, and concentration summary of an inverse-design vector.

    Parameters
    ----------
    delta_x:
        Feature-space change vector, shape ``(n_features,)``.
    threshold:
        Relative threshold (fraction of max |Δx|) for the support indicator.

    Returns
    -------
    dict with keys:
        ``support``            — indices where |Δx| > threshold × max |Δx|
        ``n_support``          — number of supported features
        ``participation_ratio``— effective feature count (‖Δx‖₁² / ‖Δx‖₂²)
        ``top3_mass``          — fraction of ‖Δx‖₁ in the 3 largest features
    """
    abs_dx = np.abs(delta_x)
    max_abs = float(abs_dx.max())
    if max_abs == 0:
        return {
            "support": np.array([], dtype=int),
            "n_support": 0,
            "participation_ratio": 0.0,
            "top3_mass": 0.0,
        }
    support = np.where(abs_dx > threshold * max_abs)[0]
    l1 = float(abs_dx.sum())
    l2sq = float((abs_dx**2).sum())
    pr = l1**2 / l2sq if l2sq > 0 else 0.0
    top3_mass = float(np.sort(abs_dx)[::-1][:3].sum()) / l1 if l1 > 0 else 0.0
    return {
        "support": support,
        "n_support": int(len(support)),
        "participation_ratio": float(pr),
        "top3_mass": float(top3_mass),
    }


# ---------------------------------------------------------------------------
# Existence-proof driver
# ---------------------------------------------------------------------------


def run_existence_proof(
    seeds: list[int] | None = None,
    base_params: LinearRecoveryParams | None = None,
) -> pd.DataFrame:
    """Run none / magnitude / orientation over multiple seeds.

    Returns a summary DataFrame with columns:
    ``manipulation``, ``delta_mean``, ``delta_std``, ``angle_mean``,
    ``angle_std``.  The ``none`` row calibrates the null floor.

    Parameters
    ----------
    seeds:
        RNG seeds to average over.  Defaults to ``list(range(10))``.
    base_params:
        Template configuration.  Defaults to ``LinearRecoveryParams()``.
    """
    if seeds is None:
        seeds = list(range(10))
    if base_params is None:
        base_params = LinearRecoveryParams()

    manipulations: list[Literal["none", "magnitude", "orientation"]] = [
        "none", "magnitude", "orientation"
    ]
    records: dict[str, list[float]] = {
        key: [] for m in manipulations for key in (f"{m}_delta", f"{m}_angle")
    }

    for seed in seeds:
        for manip in manipulations:
            params = replace(base_params, seed=seed, manipulation=manip)
            dataset = generate_dataset(params)
            delta, angle, *_ = project_and_measure(dataset, params)
            records[f"{manip}_delta"].append(delta)
            records[f"{manip}_angle"].append(angle)

    rows = []
    for manip in manipulations:
        d = records[f"{manip}_delta"]
        a = records[f"{manip}_angle"]
        rows.append(
            {
                "manipulation": manip,
                "delta_mean": float(np.mean(d)),
                "delta_std": float(np.std(d)),
                "angle_mean": float(np.mean(a)),
                "angle_std": float(np.std(a)),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_latent_trajectories(
    seed: int | None = None,
    base_params: LinearRecoveryParams | None = None,
    show_samples: bool = True,
) -> Figure:
    """3-panel PCA trajectory plot for none / magnitude / orientation.

    Each panel shows both groups' LS-mean trajectories in the top-2 PCA
    components of the feature matrix, rendered via
    ``motco.viz.plot_trajectory_from_data``.  The subplot title annotates the
    delta and angle measured in the same 2-component latent space.

    A fixed seed is shared across all panels so only the manipulation differs;
    group A's baseline trajectory is identical in all three panels.

    Parameters
    ----------
    seed:
        RNG seed for all three panels.  Defaults to ``base_params.seed``.
    base_params:
        Template configuration.  Defaults to ``LinearRecoveryParams()``.
    show_samples:
        Overlay individual sample scatter behind the trajectories.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from motco.viz import plot_trajectory_from_data

    if base_params is None:
        base_params = LinearRecoveryParams()
    if seed is None:
        seed = base_params.seed

    manipulations: list[Literal["none", "magnitude", "orientation"]] = [
        "none", "magnitude", "orientation"
    ]
    titles = {
        "none": "none — null",
        "magnitude": f"magnitude (c = {base_params.scale_c})",
        "orientation": f"orientation (θ = {base_params.angle_theta}°)",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, manip in zip(axes, manipulations):
        vis_params = replace(base_params, seed=seed, manipulation=manip, n_components=2)
        dataset = generate_dataset(vis_params)
        delta, angle, *_ = project_and_measure(dataset, vis_params)

        plot_trajectory_from_data(
            dataset.X,
            dataset.metadata,
            group_col="group",
            level_col="stage",
            full=True,
            n_components=2,
            ax=ax,
            show_samples=show_samples,
        )
        ax.set_title(f"{titles[manip]}\nδ = {delta:.2f},  θ = {angle:.1f}°", fontsize=10)

    fig.suptitle(
        "Rung 0 — Gaussian Existence Proof: Latent-space Trajectories",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    return fig
