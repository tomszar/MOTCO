3"""Cross-omic coupling test bed for trajectory recovery (Rung 4).

Adds exactly one factor on top of the Rung-3 multi-block floor: **cross-omic
coupling** — the nuisance block's per-group mean shift is derived from the anchor
block's step vector via a synthetic sparse incidence matrix M, mirroring the
InterSIM incidence-map mechanism (``generator.derive_coupled_indicators``).

In the production generator a gene is differential when any of its mapped CpGs is
(``incidence_cpg_gene.T @ indicators_methyl > 0``), introducing *structured,
direction-carrying signal* in the expression block that is a linear image of the
methylation block's differential support.  The coupling is a linear map from anchor
feature space to nuisance feature space via M, and its geometry is governed by
``MᵀM``.  When M is not an isometry (unequal singular values), the joint-space
angle between group trajectories deviates from the true anchor angle:

    cos(θ_joint) = [a·b + γ²·aᵀ(MᵀM)b] / [‖[a; γ M@a]‖ · ‖[b; γ M@b]‖]

where γ = ``coupling_scale``, a = step_A, b = step_B.  At γ=0 this is the Rung-3
independent-nuisance floor.  At γ>0, directions amplified by M's large singular
values are over-weighted in the joint inner product, distorting the measured angle.

Rung 4 sweeps ``coupling_scale`` (primary) and M structure (random sparse / dense /
rank-1) to determine whether this mechanism produces the magnitude→orientation
cross-talk the specificity study found.  A closed-form analytic prediction per seed
is compared to the PCA-measured angle to confirm coupling geometry as the mechanism.

The anchor block, per-block z-score + concatenate + PCA transform, and estimator
path (``get_model_matrix``/``build_ls_means``/``estimate_difference``) are identical
to Rungs 0–3.  ``coupling_scale = 0`` reproduces the Rung-3 independent-nuisance
baseline exactly.

This is Rung 4 of a ladder:
- Rung 0 (``linear_recovery``)       — proved the linear floor is clean.
- Rung 1 (``methylation_recovery``)  — showed ``rev.logit`` is inverted by M-value
                                        integration; not a cross-talk source.
- Rung 2 (``projector_recovery``)    — per-feature standardisation clean; SNF leaks.
- Rung 3 (``multiblock_recovery``)   — multi-block concatenation clean under
                                        independent nuisance blocks.
- Rung 4 (this module)               — adds cross-omic coupling via M and asks
                                        whether joint-space geometry distortion
                                        manufactures magnitude→orientation cross-talk.

References
----------
openspec/changes/rung4-cross-omic-coupling/design.md
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
    "CouplingRecoveryError",
    "MStructure",
    "CouplingRecoveryParams",
    "CouplingRecoveryDataset",
    "build_coupling_matrix",
    "predict_joint_angle",
    "generate_dataset",
    "project_and_measure",
    "run_coupling_sweep",
    "run_analytic_comparison",
    "run_dim_ratio_sweep",
    "run_matrix_seed_sweep",
    "plot_coupling_sweep",
    "plot_analytic_comparison",
]


class CouplingRecoveryError(ValueError):
    """Raised on invalid ``CouplingRecoveryParams``."""


MStructure = Literal["random_sparse", "dense", "rank1"]
_M_STRUCTURES: list[MStructure] = ["random_sparse", "dense", "rank1"]
_MANIPULATIONS: list[Literal["none", "magnitude", "orientation"]] = [
    "none",
    "magnitude",
    "orientation",
]
_DEFAULT_COUPLING_SCALES: list[float] = [0.0, 0.25, 0.5, 0.75, 1.0]


# ---------------------------------------------------------------------------
# Parameters and dataset
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CouplingRecoveryParams:
    """Frozen configuration for a Rung-4 cross-omic coupling run.

    The anchor block carries the known trajectory geometry (identical to
    Rungs 0–3).  The nuisance block's per-group mean is
    ``coupling_scale × M_norm @ anchor_step_g``, where
    ``M_norm = M / σ_max(M)`` is the operator-norm-normalised coupling matrix.

    Parameters
    ----------
    seed:
        RNG seed for anchor block and nuisance noise generation.
    n_features_anchor:
        Feature dimensionality of the anchor (geometry-carrying) block.
    n_samples_per_cell:
        Samples drawn per (group, stage) cell; total = 4 × this.
    noise_scale:
        Standard deviation σ applied to both anchor and nuisance noise.
    signal_scale:
        ‖a_feat‖ — the magnitude of group A's anchor-block step.
    manipulation:
        Geometric transform applied to group B's step vector.
    scale_c:
        Magnitude scale for the ``"magnitude"`` manipulation.
    angle_theta:
        Rotation angle in degrees for the ``"orientation"`` manipulation.
    dim_ratio:
        Nuisance-block dimensionality relative to the anchor:
        ``p_nuisance = max(1, round(dim_ratio × n_features_anchor))``.
    coupling_scale:
        Coupling strength γ ∈ [0, 1].  ``0`` reproduces the Rung-3
        independent-nuisance baseline (nuisance has zero mean).  ``1`` is
        maximum coupling: ‖nuisance step‖ ≤ ‖anchor step‖.
    m_structure:
        Structure of the coupling matrix M:

        - ``"random_sparse"`` — each nuisance feature independently links to
          ``nnz_per_nuis`` anchor features drawn uniformly at random
          (production-realistic; headline).
        - ``"dense"``         — M is all-ones; every nuisance feature links
          to every anchor feature (maximum-distortion upper bound).
        - ``"rank1"``         — all nuisance features link only to anchor
          feature 0 (isolated single-direction coupling).
    nnz_per_nuis:
        Expected non-zeros per row of M for the ``"random_sparse"`` structure.
        Ignored by ``"dense"`` and ``"rank1"``.
    matrix_seed:
        RNG seed for M construction.  Independent of ``seed`` so the coupling
        matrix structure and the anchor geometry are varied separately.
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
    dim_ratio: float = 1.0
    coupling_scale: float = 0.5
    m_structure: MStructure = "random_sparse"
    nnz_per_nuis: int = 3
    matrix_seed: int = 0
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
        """Feature count of the coupled nuisance block."""
        return max(1, round(self.dim_ratio * self.n_features_anchor))

    @property
    def n_features_total(self) -> int:
        """Total feature count in the joint (concatenated) matrix."""
        return self.n_features_anchor + self.n_features_nuisance

    @property
    def n_samples(self) -> int:
        """Total number of samples across all (group, stage) cells."""
        return 4 * self.n_samples_per_cell


@dataclass
class CouplingRecoveryDataset:
    """Output of :func:`generate_dataset`.

    Attributes
    ----------
    X_anchor:
        Anchor feature matrix (n_samples × n_features_anchor) in M-value space.
    X_nuisance:
        Single-element list containing the coupled nuisance block
        (n_samples × n_features_nuisance).  The list form is compatible with
        :func:`~motco.simulations.multiblock_recovery.build_joint_matrix`.
    metadata:
        Sample metadata with ``"group"`` (A/B) and ``"stage"`` (0/1) columns.
    step_A:
        Ground-truth anchor-block step for group A (μ_{A,1} − μ_{A,0}).
    step_B:
        Ground-truth anchor-block step for group B.
    M_norm:
        Operator-norm-normalised coupling matrix, shape
        ``(n_features_nuisance × n_features_anchor)``.  Used by
        :func:`predict_joint_angle` to compute the analytic prediction.
    """

    X_anchor: np.ndarray
    X_nuisance: list[np.ndarray]
    metadata: pd.DataFrame
    step_A: np.ndarray
    step_B: np.ndarray
    M_norm: np.ndarray


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate(p: CouplingRecoveryParams) -> None:
    if p.n_samples_per_cell < 2:
        raise CouplingRecoveryError("n_samples_per_cell must be >= 2")
    if p.n_components < 2:
        raise CouplingRecoveryError("n_components must be >= 2")
    if p.n_components > p.n_features_total:
        raise CouplingRecoveryError(
            f"n_components ({p.n_components}) must be <= n_features_total "
            f"({p.n_features_total})"
        )
    if p.n_components >= p.n_samples:
        raise CouplingRecoveryError(
            f"n_components ({p.n_components}) must be < n_samples ({p.n_samples})"
        )
    if p.noise_scale <= 0:
        raise CouplingRecoveryError("noise_scale must be > 0")
    if p.signal_scale <= 0:
        raise CouplingRecoveryError("signal_scale must be > 0")
    if p.dim_ratio <= 0:
        raise CouplingRecoveryError("dim_ratio must be > 0")
    if not (0.0 <= p.coupling_scale <= 1.0):
        raise CouplingRecoveryError("coupling_scale must be in [0, 1]")
    if p.m_structure not in _M_STRUCTURES:
        raise CouplingRecoveryError(
            f"m_structure must be one of {_M_STRUCTURES}; got {p.m_structure!r}"
        )
    if p.m_structure == "random_sparse":
        if p.nnz_per_nuis < 1:
            raise CouplingRecoveryError(
                "nnz_per_nuis must be >= 1 for random_sparse structure"
            )
        if p.nnz_per_nuis > p.n_features_anchor:
            raise CouplingRecoveryError(
                f"nnz_per_nuis ({p.nnz_per_nuis}) must be <= n_features_anchor "
                f"({p.n_features_anchor})"
            )
    if p.manipulation not in ("none", "magnitude", "orientation"):
        raise CouplingRecoveryError(
            f"manipulation must be 'none', 'magnitude', or 'orientation'; "
            f"got {p.manipulation!r}"
        )


# ---------------------------------------------------------------------------
# Coupling matrix construction
# ---------------------------------------------------------------------------


def build_coupling_matrix(
    p_nuis: int,
    p_anchor: int,
    structure: MStructure,
    nnz_per_nuis: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build and operator-norm-normalise a coupling matrix M.

    Constructs M ∈ {0,1}^{p_nuis × p_anchor} according to ``structure``, then
    normalises by the largest singular value so ``‖M_norm‖₂ = 1``.

    Parameters
    ----------
    p_nuis:
        Number of nuisance features (rows of M).
    p_anchor:
        Number of anchor features (columns of M).
    structure:
        ``"random_sparse"`` — each row links to ``nnz_per_nuis`` anchor
        features chosen uniformly without replacement.
        ``"dense"``          — all-ones matrix.
        ``"rank1"``          — all rows link only to anchor feature 0.
    nnz_per_nuis:
        Non-zeros per row for ``"random_sparse"``; ignored otherwise.
    rng:
        RNG for ``"random_sparse"`` column sampling.

    Returns
    -------
    M_norm : np.ndarray, shape (p_nuis, p_anchor)
        Operator-norm-normalised coupling matrix (``‖M_norm‖₂ = 1``).
    """
    if structure == "dense":
        M = np.ones((p_nuis, p_anchor), dtype=float)
    elif structure == "rank1":
        M = np.zeros((p_nuis, p_anchor), dtype=float)
        M[:, 0] = 1.0
    else:  # random_sparse
        M = np.zeros((p_nuis, p_anchor), dtype=float)
        n_links = min(nnz_per_nuis, p_anchor)
        for i in range(p_nuis):
            cols = rng.choice(p_anchor, size=n_links, replace=False)
            M[i, cols] = 1.0

    sigma_max = float(np.linalg.svd(M, compute_uv=False)[0])
    return M / sigma_max


# ---------------------------------------------------------------------------
# Analytic prediction
# ---------------------------------------------------------------------------


def predict_joint_angle(
    step_A: np.ndarray,
    step_B: np.ndarray,
    M_norm: np.ndarray,
    coupling_scale: float,
) -> float:
    """Analytic joint-space angle prediction (pre-PCA, joint feature space).

    Computes the angle between the joint steps ``[step_A; γ·M@step_A]`` and
    ``[step_B; γ·M@step_B]`` using the closed-form expression:

        cos(θ) = [a·b + γ²·(M@a)·(M@b)] / [‖[a; γ M@a]‖ · ‖[b; γ M@b]‖]

    where a = ``step_A``, b = ``step_B``, γ = ``coupling_scale``.

    At γ = 0 this equals the anchor-space angle between step_A and step_B.
    For ``"magnitude"`` (b = c·a), the numerator equals c × ‖[a; γ M@a]‖²
    and the denominator equals c × ‖[a; γ M@a]‖², so cos(θ) = 1 (θ = 0°)
    for any M and γ — the coupling is direction-preserving for uniform scaling.

    Parameters
    ----------
    step_A:
        Anchor-block step vector for group A, shape (p_anchor,).
    step_B:
        Anchor-block step vector for group B, shape (p_anchor,).
    M_norm:
        Operator-norm-normalised coupling matrix, shape (p_nuis, p_anchor).
    coupling_scale:
        Coupling strength γ ∈ [0, 1].

    Returns
    -------
    float
        Predicted angle in degrees.
    """
    gamma = coupling_scale
    Ma = M_norm @ step_A
    Mb = M_norm @ step_B
    numerator = float(np.dot(step_A, step_B) + gamma**2 * np.dot(Ma, Mb))
    norm_A = float(np.sqrt(np.dot(step_A, step_A) + gamma**2 * np.dot(Ma, Ma)))
    norm_B = float(np.sqrt(np.dot(step_B, step_B) + gamma**2 * np.dot(Mb, Mb)))
    denom = norm_A * norm_B
    if denom < 1e-12:
        return 0.0
    cos_theta = np.clip(numerator / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------


def generate_dataset(params: CouplingRecoveryParams) -> CouplingRecoveryDataset:
    """Generate anchor + coupled nuisance blocks with known anchor geometry.

    The anchor block step vectors are constructed by the Rung-0 generator
    (reused unchanged).  The anchor block noise is drawn identically to Rung 3
    (fresh RNG seeded by ``params.seed``; isotropic, no anisotropy).

    The nuisance block mean for group g, stage s is:
    ``coupling_scale × M_norm @ step_g`` (zero at stage 0 for both groups).
    Independent Gaussian noise with ``noise_scale`` σ is added to every nuisance
    sample.  At ``coupling_scale = 0``, the nuisance block is pure noise, exactly
    reproducing the Rung-3 ``rho_nuisance = 0, dim_ratio`` baseline.

    The coupling matrix M is built with ``matrix_seed`` independently of
    ``seed``, so anchor geometry and M structure can be swept separately.

    Parameters
    ----------
    params:
        Generation configuration.
    """
    _validate(params)

    lin = _linear_generate_dataset(params.as_linear())
    step_A, step_B = lin.step_A, lin.step_B

    p_a = params.n_features_anchor
    p_n = params.n_features_nuisance
    n = params.n_samples_per_cell

    rng_m = np.random.default_rng(params.matrix_seed)
    M_norm = build_coupling_matrix(
        p_n, p_a, params.m_structure, params.nnz_per_nuis, rng_m
    )

    anchor_mu: dict[tuple[str, str], np.ndarray] = {
        ("A", "0"): np.zeros(p_a),
        ("A", "1"): step_A,
        ("B", "0"): np.zeros(p_a),
        ("B", "1"): step_B,
    }
    nuis_mu: dict[tuple[str, str], np.ndarray] = {
        ("A", "0"): np.zeros(p_n),
        ("A", "1"): params.coupling_scale * M_norm @ step_A,
        ("B", "0"): np.zeros(p_n),
        ("B", "1"): params.coupling_scale * M_norm @ step_B,
    }

    rng = np.random.default_rng(params.seed)

    anchor_rows: list[np.ndarray] = []
    nuis_rows: list[np.ndarray] = []
    meta_rows: list[tuple[str, str]] = []
    for group in ("A", "B"):
        for stage in ("0", "1"):
            anchor_rows.append(
                anchor_mu[(group, stage)]
                + rng.standard_normal((n, p_a)) * params.noise_scale
            )
            nuis_rows.append(
                nuis_mu[(group, stage)]
                + rng.standard_normal((n, p_n)) * params.noise_scale
            )
            meta_rows.extend([(group, stage)] * n)

    metadata = pd.DataFrame(meta_rows, columns=["group", "stage"])
    return CouplingRecoveryDataset(
        X_anchor=np.vstack(anchor_rows),
        X_nuisance=[np.vstack(nuis_rows)],
        metadata=metadata,
        step_A=step_A,
        step_B=step_B,
        M_norm=M_norm,
    )


# ---------------------------------------------------------------------------
# Joint matrix, projection, and measurement
# ---------------------------------------------------------------------------


def _build_joint_matrix(dataset: CouplingRecoveryDataset) -> tuple[np.ndarray, int]:
    """Per-block z-score and concatenate anchor + nuisance.

    Applies the production ``concat`` transform (``std == 0 → 1``, matching
    ``evaluation.py:_concat_integration``) identically to Rung 3.
    """
    blocks = [dataset.X_anchor] + dataset.X_nuisance
    standardised: list[np.ndarray] = []
    for X in blocks:
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True)
        std[std == 0.0] = 1.0
        standardised.append((X - mean) / std)
    return np.hstack(standardised), dataset.X_anchor.shape[1]


def _measure_projected(
    Y: np.ndarray,
    metadata: pd.DataFrame,
    n_components: int,
) -> tuple[float, float]:
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
    dataset: CouplingRecoveryDataset,
    params: CouplingRecoveryParams,
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
    X_joint, _ = _build_joint_matrix(dataset)
    pca = PCA(n_components=params.n_components)
    Y = pca.fit_transform(X_joint)
    delta, angle = _measure_projected(Y, dataset.metadata, params.n_components)
    return delta, angle, Y


# ---------------------------------------------------------------------------
# Drivers / sweeps
# ---------------------------------------------------------------------------


def _measure_over_seeds(
    base_params: CouplingRecoveryParams,
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


def run_coupling_sweep(
    coupling_scales: list[float] | None = None,
    m_structures: list[MStructure] | None = None,
    seeds: list[int] | None = None,
    base_params: CouplingRecoveryParams | None = None,
) -> pd.DataFrame:
    """Primary sweep: coupling_scale × M structure × manipulations.

    ``coupling_scale = 0`` is the Rung-3 independent-nuisance baseline.  The
    ``magnitude`` arm serves as a within-run no-cross-talk control: the analytic
    formula proves its angle must stay at the ``none`` floor for any M and γ.

    Parameters
    ----------
    coupling_scales:
        Defaults to ``[0.0, 0.25, 0.5, 0.75, 1.0]``.
    m_structures:
        Defaults to all three structures.
    seeds:
        Defaults to ``list(range(10))``.
    base_params:
        Template configuration.

    Returns
    -------
    pd.DataFrame
        One row per ``(m_structure, coupling_scale, manipulation)`` with
        ``delta_mean``, ``delta_std``, ``angle_mean``, ``angle_std``.
    """
    if coupling_scales is None:
        coupling_scales = _DEFAULT_COUPLING_SCALES
    if m_structures is None:
        m_structures = list(_M_STRUCTURES)
    if seeds is None:
        seeds = list(range(10))
    if base_params is None:
        base_params = CouplingRecoveryParams()

    rows = []
    for struct in m_structures:
        for gamma in coupling_scales:
            for manip in _MANIPULATIONS:
                deltas, angles = _measure_over_seeds(
                    base_params, manip, seeds,
                    m_structure=struct, coupling_scale=gamma,
                )
                rows.append(
                    {
                        "m_structure": struct,
                        "coupling_scale": gamma,
                        "manipulation": manip,
                        "delta_mean": float(np.mean(deltas)),
                        "delta_std": float(np.std(deltas)),
                        "angle_mean": float(np.mean(angles)),
                        "angle_std": float(np.std(angles)),
                    }
                )
    return pd.DataFrame(rows)


def run_analytic_comparison(
    coupling_scales: list[float] | None = None,
    m_structures: list[MStructure] | None = None,
    seeds: list[int] | None = None,
    base_params: CouplingRecoveryParams | None = None,
) -> pd.DataFrame:
    """Compare analytic joint-space angle prediction to PCA-measured angle.

    For each (seed, coupling_scale, M structure), computes:
    - ``angle_pred``  — closed-form feature-space prediction from
      :func:`predict_joint_angle`.
    - ``angle_meas``  — empirical PCA-measured angle.
    - ``angle_anchor`` — baseline angle measured with ``coupling_scale = 0``
      (same seed; approximates the true anchor angle under PCA projection).

    Only the ``"orientation"`` manipulation is included (the ``"magnitude"``
    arm has ``angle_pred = 0°`` analytically and serves as a separate control).

    Parameters
    ----------
    coupling_scales:
        Defaults to ``[0.0, 0.25, 0.5, 0.75, 1.0]``.
    m_structures:
        Defaults to all three structures.
    seeds:
        Defaults to ``list(range(10))``.
    base_params:
        Template configuration.

    Returns
    -------
    pd.DataFrame
        One row per ``(m_structure, coupling_scale, seed)`` with
        ``angle_pred``, ``angle_meas``, ``angle_anchor``.
    """
    if coupling_scales is None:
        coupling_scales = _DEFAULT_COUPLING_SCALES
    if m_structures is None:
        m_structures = list(_M_STRUCTURES)
    if seeds is None:
        seeds = list(range(10))
    if base_params is None:
        base_params = CouplingRecoveryParams()

    rows = []
    for struct in m_structures:
        for seed in seeds:
            # Baseline angle (coupling_scale=0, same seed) — approximates true
            # anchor angle after PCA projection.
            p_base = replace(
                base_params,
                seed=seed,
                manipulation="orientation",
                m_structure=struct,
                coupling_scale=0.0,
            )
            ds_base = generate_dataset(p_base)
            _, angle_anchor, _ = project_and_measure(ds_base, p_base)

            for gamma in coupling_scales:
                p = replace(
                    base_params,
                    seed=seed,
                    manipulation="orientation",
                    m_structure=struct,
                    coupling_scale=gamma,
                )
                ds = generate_dataset(p)
                _, angle_meas, _ = project_and_measure(ds, p)
                angle_pred = predict_joint_angle(
                    ds.step_A, ds.step_B, ds.M_norm, gamma
                )
                rows.append(
                    {
                        "m_structure": struct,
                        "coupling_scale": gamma,
                        "seed": seed,
                        "angle_pred": angle_pred,
                        "angle_meas": angle_meas,
                        "angle_anchor": angle_anchor,
                    }
                )
    return pd.DataFrame(rows)


def run_dim_ratio_sweep(
    dim_ratios: list[float] | None = None,
    coupling_scale: float = 0.75,
    seeds: list[int] | None = None,
    base_params: CouplingRecoveryParams | None = None,
) -> pd.DataFrame:
    """Secondary sweep: dim_ratio at a fixed coupling_scale (random_sparse).

    Tests whether orientation distortion scales with the nuisance block's
    weight in the joint PCA, as expected from the analytic formula.

    Parameters
    ----------
    dim_ratios:
        Defaults to ``[0.5, 1.0, 5.0]``.
    coupling_scale:
        Fixed coupling strength.  Defaults to ``0.75``.
    seeds:
        Defaults to ``list(range(10))``.
    base_params:
        Template configuration.

    Returns
    -------
    pd.DataFrame
        One row per ``(dim_ratio, manipulation)`` with delta/angle stats.
    """
    if dim_ratios is None:
        dim_ratios = [0.5, 1.0, 5.0]
    if seeds is None:
        seeds = list(range(10))
    if base_params is None:
        base_params = CouplingRecoveryParams()

    rows = []
    for dr in dim_ratios:
        for manip in _MANIPULATIONS:
            deltas, angles = _measure_over_seeds(
                base_params, manip, seeds,
                m_structure="random_sparse",
                coupling_scale=coupling_scale,
                dim_ratio=dr,
            )
            rows.append(
                {
                    "dim_ratio": dr,
                    "coupling_scale": coupling_scale,
                    "manipulation": manip,
                    "delta_mean": float(np.mean(deltas)),
                    "delta_std": float(np.std(deltas)),
                    "angle_mean": float(np.mean(angles)),
                    "angle_std": float(np.std(angles)),
                }
            )
    return pd.DataFrame(rows)


def run_matrix_seed_sweep(
    matrix_seeds: list[int] | None = None,
    coupling_scale: float = 0.75,
    seeds: list[int] | None = None,
    base_params: CouplingRecoveryParams | None = None,
) -> pd.DataFrame:
    """Stability check: vary matrix_seed at fixed coupling_scale.

    Confirms that results are not artefacts of the particular M realisation
    drawn for the headline run.

    Parameters
    ----------
    matrix_seeds:
        Defaults to ``list(range(5))``.
    coupling_scale:
        Fixed coupling strength.  Defaults to ``0.75``.
    seeds:
        Defaults to ``list(range(10))``.
    base_params:
        Template configuration.

    Returns
    -------
    pd.DataFrame
        One row per ``(matrix_seed, manipulation)`` with delta/angle stats.
    """
    if matrix_seeds is None:
        matrix_seeds = list(range(5))
    if seeds is None:
        seeds = list(range(10))
    if base_params is None:
        base_params = CouplingRecoveryParams()

    rows = []
    for ms in matrix_seeds:
        for manip in _MANIPULATIONS:
            deltas, angles = _measure_over_seeds(
                base_params, manip, seeds,
                matrix_seed=ms,
                coupling_scale=coupling_scale,
                m_structure="random_sparse",
            )
            rows.append(
                {
                    "matrix_seed": ms,
                    "coupling_scale": coupling_scale,
                    "manipulation": manip,
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


def plot_coupling_sweep(sweep: pd.DataFrame) -> Figure:
    """Line plot of delta/angle vs coupling_scale per M structure.

    Three panels per statistic (delta, angle), one per M structure.  Each
    panel shows three lines (none / magnitude / orientation) with ±1 SD bands.

    Parameters
    ----------
    sweep:
        Output of :func:`run_coupling_sweep`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    structures = [s for s in _M_STRUCTURES if s in sweep["m_structure"].values]
    colors = {"none": "0.5", "magnitude": "C0", "orientation": "C1"}

    fig, axes = plt.subplots(
        2, len(structures), figsize=(5 * len(structures), 9), squeeze=False
    )

    for col, struct in enumerate(structures):
        sub = sweep[sweep["m_structure"] == struct]
        for row, stat in enumerate(("delta", "angle")):
            ax = axes[row][col]
            for manip in _MANIPULATIONS:
                s = sub[sub["manipulation"] == manip].sort_values("coupling_scale")
                x = s["coupling_scale"].to_numpy()
                y = s[f"{stat}_mean"].to_numpy()
                ye = s[f"{stat}_std"].to_numpy()
                ax.plot(x, y, marker="o", color=colors[manip], label=manip)
                ax.fill_between(x, y - ye, y + ye, alpha=0.2, color=colors[manip])
            ax.set_xlabel("coupling_scale  (γ)")
            unit = "  (degrees)" if stat == "angle" else ""
            ax.set_ylabel(f"measured {stat}{unit}")
            if row == 0:
                ax.set_title(f"M structure: {struct}")
            if col == 0:
                ax.legend(title="manipulation", fontsize=8)

    fig.suptitle(
        "Rung 4 — Cross-omic coupling: delta/angle vs coupling strength",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    return fig


def plot_analytic_comparison(comparison: pd.DataFrame) -> Figure:
    """Scatter of analytic prediction vs PCA-measured angle (orientation arm).

    One panel per M structure.  Each point is one (seed, coupling_scale) pair,
    coloured by coupling_scale.  The diagonal y=x is the perfect-prediction
    line; points above it indicate PCA adds distortion beyond the analytic
    coupling formula; points below indicate PCA partially compensates.

    Parameters
    ----------
    comparison:
        Output of :func:`run_analytic_comparison`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    structures = [s for s in _M_STRUCTURES if s in comparison["m_structure"].values]
    fig, axes = plt.subplots(1, len(structures), figsize=(5 * len(structures), 5))
    if len(structures) == 1:
        axes = [axes]

    for ax, struct in zip(axes, structures):
        sub = comparison[comparison["m_structure"] == struct]
        sc = ax.scatter(
            sub["angle_pred"],
            sub["angle_meas"],
            c=sub["coupling_scale"],
            cmap="viridis",
            s=30,
            alpha=0.7,
        )
        # Diagonal y = x
        lim_min = min(sub["angle_pred"].min(), sub["angle_meas"].min()) - 2
        lim_max = max(sub["angle_pred"].max(), sub["angle_meas"].max()) + 2
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", linewidth=0.8)
        plt.colorbar(sc, ax=ax, label="coupling_scale")
        ax.set_xlabel("analytic predicted angle (degrees)")
        ax.set_ylabel("PCA measured angle (degrees)")
        ax.set_title(f"M structure: {struct}")

    fig.suptitle(
        "Rung 4 — Analytic vs empirical angle (orientation, all seeds)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    return fig
