"""Methylation rev.logit test bed for trajectory recovery (Rung 1).

Layers InterSIM's methylation nonlinearity onto the Rung-0 geometry-injection
test bed as the *single* new variable. The known 2-stage geometry
(``none``/``magnitude``/``orientation``) is injected in methylation **M-value
(logit) space** exactly as in Rung 0, then the drawn samples are passed through
:func:`motco.simulations.generator.rev_logit` to obtain β values, projected with
inline PCA, and measured with the production ``stats/trajectory.py`` estimators.

Methylation-only: no expression/protein layer is generated or measured. The
sigmoid **operating point** (a baseline M-value offset placing the trajectory on
the ``rev_logit`` curve) is the swept independent variable — ``rev_logit`` is
locally linear at M ≈ 0 (slope ≈ 0.25) and saturates on the tails, so distortion
of ``delta``/``angle`` and the cross-talk between them are functions of where on
the sigmoid the step sits.

This is Rung 1 of a ladder. Rung 0 (``linear_recovery``) proved the linear floor
is clean; Rung 1 isolates the ``rev_logit`` nonlinearity; later rungs add the
PLS projector and the cross-omic cascade.

References
----------
openspec/changes/rung1-methylation-nonlinearity/design.md
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from motco.simulations.generator import rev_logit
from motco.simulations.linear_recovery import (
    LinearRecoveryDataset,
    LinearRecoveryParams,
    delta_x_summary,
    givens_rotation,
    inverse_design_magnitude,
    inverse_design_orientation,
)
from motco.simulations.linear_recovery import (
    generate_dataset as _linear_generate_dataset,
)
from motco.simulations.linear_recovery import (
    project_and_measure as _linear_project_and_measure,
)

__all__ = [
    "MethylationRecoveryError",
    "MethylationRecoveryParams",
    "MethylationRecoveryDataset",
    "generate_dataset",
    "project_and_measure",
    "jacobian_diag",
    "inverse_design_magnitude_mvalue",
    "inverse_design_orientation_mvalue",
    "run_operating_point_sweep",
    "run_step_scale_sweep",
    "plot_operating_point_sweep",
    # re-exported Rung-0 helpers for convenience
    "givens_rotation",
    "delta_x_summary",
]


class MethylationRecoveryError(ValueError):
    """Raised on invalid ``MethylationRecoveryParams``."""


@dataclass(frozen=True)
class MethylationRecoveryParams:
    """Frozen configuration for a Rung-1 methylation recovery run.

    Mirrors :class:`~motco.simulations.linear_recovery.LinearRecoveryParams`
    (the geometry is injected identically) and adds ``m_baseline``: a scalar
    M-value offset applied to every CpG, placing the whole trajectory at a
    chosen point on the ``rev_logit`` sigmoid.

    Parameters
    ----------
    seed:
        RNG seed for deterministic generation.
    n_features:
        Number of CpGs (M-value features).
    n_samples_per_cell:
        Samples drawn per (group, stage) cell.
    noise_scale:
        Standard deviation of isotropic Gaussian noise in M-value space.
    signal_scale:
        ‖a_feat‖ — the magnitude of group A's M-space step (logit units).
    manipulation:
        Geometric transform applied to group B's step vector.
    scale_c:
        Magnitude scale for the ``"magnitude"`` manipulation (step_B = c · a_feat).
    angle_theta:
        Rotation angle in degrees for the ``"orientation"`` manipulation.
    n_components:
        Number of PCA components retained for measurement. Defaults to ``2`` —
        the 2-group × 2-stage signal lives in ≤ 2 dimensions, and Rung 0 showed
        the angle null floor grows with each retained *noise* PC. Rung 1 probes
        a small (≈ 5–10°) cross-talk signal, so it retains only the signal
        subspace to keep the ``rev_logit`` nonlinearity — not the k-noise floor
        already characterized in Rung 0 — the variable under test.
    m_baseline:
        Scalar M-value baseline (operating point). ``0.0`` sits at the sigmoid
        center (β = 0.5, ~linear); larger magnitudes move into saturation.
    """

    seed: int = 0
    n_features: int = 50
    n_samples_per_cell: int = 40
    noise_scale: float = 0.3
    signal_scale: float = 2.0
    manipulation: Literal["none", "magnitude", "orientation"] = "none"
    scale_c: float = 2.0
    angle_theta: float = 45.0
    n_components: int = 2
    m_baseline: float = 0.0

    def as_linear(self) -> LinearRecoveryParams:
        """Project to the Rung-0 params (shared fields) for helper reuse."""
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
class MethylationRecoveryDataset:
    """Output of :func:`generate_dataset`.

    Attributes
    ----------
    X:
        Methylation β matrix (n_samples × n_features), values in (0, 1).
    metadata:
        Sample metadata with ``"group"`` (A/B) and ``"stage"`` (0/1) columns,
        row-aligned with ``X``.
    step_A:
        Ground-truth **M-space** step for group A (μ_{A,1} − μ_{A,0}).
    step_B:
        Ground-truth **M-space** step for group B.
    m_baseline:
        The scalar M-value operating point used for generation.
    """

    X: pd.DataFrame
    metadata: pd.DataFrame
    step_A: np.ndarray
    step_B: np.ndarray
    m_baseline: float


# ---------------------------------------------------------------------------
# Validation and generation
# ---------------------------------------------------------------------------


def _validate(p: MethylationRecoveryParams) -> None:
    if p.n_samples_per_cell < 2:
        raise MethylationRecoveryError("n_samples_per_cell must be >= 2")
    if p.n_components < 2:
        raise MethylationRecoveryError("n_components must be >= 2")
    if p.n_components > p.n_features:
        raise MethylationRecoveryError(
            f"n_components ({p.n_components}) must be <= n_features ({p.n_features})"
        )
    if p.noise_scale <= 0:
        raise MethylationRecoveryError("noise_scale must be > 0")
    if p.signal_scale <= 0:
        raise MethylationRecoveryError("signal_scale must be > 0")
    if not np.isfinite(p.m_baseline):
        raise MethylationRecoveryError("m_baseline must be finite")
    if p.manipulation not in ("none", "magnitude", "orientation"):
        raise MethylationRecoveryError(
            f"manipulation must be 'none', 'magnitude', or 'orientation'; "
            f"got {p.manipulation!r}"
        )


def generate_dataset(params: MethylationRecoveryParams) -> MethylationRecoveryDataset:
    """Generate a methylation β dataset with known M-space trajectory geometry.

    The per-group M-space steps (``step_A``, ``step_B``) are constructed by the
    Rung-0 generator (reused unchanged), so the injected geometry is identical to
    Rung 0. M-space means are ``μ_{g,0} = m_baseline`` and
    ``μ_{g,1} = m_baseline + step_g``; samples are drawn
    ``x_M = μ + N(0, noise_scale² I)`` and mapped to β via :func:`rev_logit`.

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
    base = np.full(p, params.m_baseline, dtype=float)
    mu: dict[tuple[str, str], np.ndarray] = {
        ("A", "0"): base,
        ("A", "1"): base + step_A,
        ("B", "0"): base,
        ("B", "1"): base + step_B,
    }

    rng = np.random.default_rng(params.seed)
    rows: list[np.ndarray] = []
    meta_rows: list[tuple[str, str]] = []
    for group in ("A", "B"):
        for stage in ("0", "1"):
            noise = rng.standard_normal((n, p)) * params.noise_scale
            m_vals = mu[(group, stage)] + noise
            rows.append(rev_logit(m_vals))
            meta_rows.extend([(group, stage)] * n)

    beta = np.vstack(rows)
    X_df = pd.DataFrame(beta, columns=[f"cg{i}" for i in range(p)])
    metadata = pd.DataFrame(meta_rows, columns=["group", "stage"])

    return MethylationRecoveryDataset(
        X=X_df,
        metadata=metadata,
        step_A=step_A,
        step_B=step_B,
        m_baseline=params.m_baseline,
    )


# ---------------------------------------------------------------------------
# Projection and measurement (reuses the Rung-0 path on the β matrix)
# ---------------------------------------------------------------------------


def project_and_measure(
    dataset: MethylationRecoveryDataset,
    params: MethylationRecoveryParams,
) -> tuple[float, float, object, pd.DataFrame, np.ndarray]:
    """Fit PCA on β, project, and measure delta/angle via the production path.

    Delegates to :func:`motco.simulations.linear_recovery.project_and_measure`
    with the β matrix as the feature matrix, so the projection (inline PCA, no
    per-feature standardization) and measurement (``estimate_difference`` on a
    2-group × 2-stage design) are exactly the Rung-0 code path.

    Returns
    -------
    delta, angle, pca, Y, Vk
        See :func:`motco.simulations.linear_recovery.project_and_measure`.
    """
    lin_ds = LinearRecoveryDataset(
        X=dataset.X,
        metadata=dataset.metadata,
        step_A=dataset.step_A,
        step_B=dataset.step_B,
    )
    return _linear_project_and_measure(lin_ds, params.as_linear())


# ---------------------------------------------------------------------------
# First-order (Jacobian) inverse design at the operating point
# ---------------------------------------------------------------------------


def jacobian_diag(m_baseline: float, n_features: int) -> np.ndarray:
    """Diagonal of the M→β Jacobian at a scalar operating point.

    The local derivative of ``rev_logit`` is ``β(1 − β)``; at a uniform baseline
    every CpG shares the same slope, so this returns a length-``n_features``
    vector of that constant.
    """
    b = rev_logit(np.full(n_features, m_baseline, dtype=float))
    return b * (1.0 - b)


def inverse_design_magnitude_mvalue(
    a: np.ndarray, Vk: np.ndarray, c: float, m_baseline: float
) -> np.ndarray:
    """First-order M-space change achieving a magnitude-c latent step.

    Composes the Rung-0 exact inverse design (which yields the β-feature
    preimage ``Δβ = Vk·(c−1)·a``) with the inverse Jacobian at the operating
    point, ``Δm = Δβ / (β(1−β))``. This is a *first-order* preimage: the
    nonlinear forward ``rev_logit(m_baseline + Δm) − rev_logit(m_baseline)``
    recovers ``Δβ`` only near the operating point and degrades into saturation.
    """
    d_beta = inverse_design_magnitude(a, Vk, c)
    j = jacobian_diag(m_baseline, Vk.shape[0])
    return d_beta / j


def inverse_design_orientation_mvalue(
    a: np.ndarray, Vk: np.ndarray, R: np.ndarray, m_baseline: float
) -> np.ndarray:
    """First-order M-space change achieving an R-rotated latent step.

    As :func:`inverse_design_magnitude_mvalue`, but for the orientation target
    ``Δβ = Vk·(R−I)·a``. First-order; see that function's caveat.
    """
    d_beta = inverse_design_orientation(a, Vk, R)
    j = jacobian_diag(m_baseline, Vk.shape[0])
    return d_beta / j


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------

_MANIPULATIONS: list[Literal["none", "magnitude", "orientation"]] = [
    "none",
    "magnitude",
    "orientation",
]


def _measure_over_seeds(
    base_params: MethylationRecoveryParams,
    manip: Literal["none", "magnitude", "orientation"],
    seeds: list[int],
    **overrides: Any,
) -> tuple[list[float], list[float]]:
    deltas: list[float] = []
    angles: list[float] = []
    for seed in seeds:
        params = replace(base_params, seed=seed, manipulation=manip, **overrides)
        dataset = generate_dataset(params)
        delta, angle, *_ = project_and_measure(dataset, params)
        deltas.append(delta)
        angles.append(angle)
    return deltas, angles


def run_operating_point_sweep(
    m_baselines: list[float],
    seeds: list[int] | None = None,
    base_params: MethylationRecoveryParams | None = None,
) -> pd.DataFrame:
    """Sweep the sigmoid operating point for all three manipulations.

    Returns a tidy DataFrame with one row per (operating point, manipulation):
    ``m_baseline``, ``beta_baseline`` (= ``rev_logit(m_baseline)``), ``slope``
    (= ``β(1−β)``, the local sigmoid gain), ``manipulation``, and the
    mean ± SD of ``delta``/``angle`` over ``seeds``. The ``none`` rows give the
    per-operating-point null floor against which cross-talk is read.

    Parameters
    ----------
    m_baselines:
        Operating points (scalar M-value baselines) to sweep.
    seeds:
        RNG seeds to average over. Defaults to ``list(range(10))``.
    base_params:
        Template configuration. Defaults to ``MethylationRecoveryParams()``.
    """
    if seeds is None:
        seeds = list(range(10))
    if base_params is None:
        base_params = MethylationRecoveryParams()

    rows = []
    for m_baseline in m_baselines:
        b = float(rev_logit(np.array([m_baseline]))[0])
        for manip in _MANIPULATIONS:
            deltas, angles = _measure_over_seeds(
                base_params, manip, seeds, m_baseline=m_baseline
            )
            rows.append(
                {
                    "m_baseline": float(m_baseline),
                    "beta_baseline": b,
                    "slope": b * (1.0 - b),
                    "manipulation": manip,
                    "delta_mean": float(np.mean(deltas)),
                    "delta_std": float(np.std(deltas)),
                    "angle_mean": float(np.mean(angles)),
                    "angle_std": float(np.std(angles)),
                }
            )
    return pd.DataFrame(rows)


def run_step_scale_sweep(
    signal_scales: list[float],
    seeds: list[int] | None = None,
    base_params: MethylationRecoveryParams | None = None,
) -> pd.DataFrame:
    """Sweep the step scale at a fixed (center) operating point.

    Separates distortion driven by *step span* across the sigmoid from
    distortion driven by *baseline offset*. ``m_baseline`` is held at
    ``base_params.m_baseline``. Returns the same column layout as
    :func:`run_operating_point_sweep`, with an added ``signal_scale`` column.

    Parameters
    ----------
    signal_scales:
        M-space step magnitudes ‖a_feat‖ to sweep.
    seeds:
        RNG seeds to average over. Defaults to ``list(range(10))``.
    base_params:
        Template configuration. Defaults to ``MethylationRecoveryParams()``.
    """
    if seeds is None:
        seeds = list(range(10))
    if base_params is None:
        base_params = MethylationRecoveryParams()

    b = float(rev_logit(np.array([base_params.m_baseline]))[0])
    rows = []
    for scale in signal_scales:
        for manip in _MANIPULATIONS:
            deltas, angles = _measure_over_seeds(
                base_params, manip, seeds, signal_scale=scale
            )
            rows.append(
                {
                    "signal_scale": float(scale),
                    "m_baseline": float(base_params.m_baseline),
                    "beta_baseline": b,
                    "slope": b * (1.0 - b),
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


def plot_operating_point_sweep(sweep: pd.DataFrame) -> Figure:
    """Plot delta and angle versus operating point for each manipulation.

    Two panels (delta, angle) versus ``m_baseline`` with one line per
    manipulation and a secondary axis showing the sigmoid slope ``β(1−β)``.

    Parameters
    ----------
    sweep:
        Output of :func:`run_operating_point_sweep`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"none": "0.6", "magnitude": "C0", "orientation": "C1"}

    for stat, ax in zip(("delta", "angle"), axes):
        for manip in _MANIPULATIONS:
            sub = sweep[sweep["manipulation"] == manip].sort_values("m_baseline")
            ax.errorbar(
                sub["m_baseline"],
                sub[f"{stat}_mean"],
                yerr=sub[f"{stat}_std"],
                marker="o",
                capsize=3,
                color=colors[manip],
                label=manip,
            )
        ax.set_xlabel("operating point  m_baseline  (M-value)")
        ax.set_ylabel(f"measured {stat}" + ("  (degrees)" if stat == "angle" else ""))
        ax.set_title(f"{stat} vs operating point")
        ax.legend(title="manipulation", fontsize=8)

        slope_src = sweep.drop_duplicates("m_baseline").sort_values("m_baseline")
        ax2 = ax.twinx()
        ax2.plot(
            slope_src["m_baseline"],
            slope_src["slope"],
            color="0.3",
            ls="--",
            lw=1,
            alpha=0.6,
        )
        ax2.set_ylabel("sigmoid slope  β(1−β)", color="0.3", fontsize=8)
        ax2.tick_params(axis="y", labelsize=7, colors="0.3")

    fig.suptitle(
        "Rung 1 — Methylation rev.logit: recovery vs operating point",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    return fig
