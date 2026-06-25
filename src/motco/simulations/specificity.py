"""Descriptive specificity instrumentation for the feature-surgery modes.

Characterizes how MOTCO *responds* to each realistic trajectory difference —
which statistics move and how much they cross-talk — rather than gating on a
clean diagonal. For each mode it generates several replicates, evaluates them
through the MOTCO trajectory test with RRPP, and reports the per-statistic
rejection rate plus a group-vs-stage projection diagnostic (how much of the
injected group signal lands in the disease/stage-discriminant subspace).

This is a *descriptive* tool, not a pass/fail gate: cross-talk (e.g. magnitude
bending shape via the methylation ``rev.logit`` nonlinearity) and even
non-detection are findings, not failures. The heavier cluster-run study
produces the definitive specificity matrix and power curves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from motco.simulations.evaluation import (
    SimulationEvaluationParams,
    evaluate_semisynthetic_trajectory,
    integrate_semisynthetic_dataset,
)
from motco.simulations.reference import IntersimReference, load_reference
from motco.simulations.semisynthetic import (
    SemiSyntheticTrajectoryParams,
    TrajectoryMode,
    generate_semisynthetic_trajectory,
)

STATISTICS: tuple[str, ...] = ("delta", "angle", "shape")

#: The statistic each non-null mode is designed to move.
TARGET_STATISTIC: dict[str, str] = {
    "magnitude": "delta",
    "orientation": "angle",
    "shape": "shape",
}


#: Shape is degenerate with a single trajectory step, so the 2-stage isolation
#: pass only exercises the size/orientation modes plus the negative controls.
SHAPE_FREE_MODES: tuple[str, ...] = ("none", "translation", "magnitude", "orientation")


@dataclass(frozen=True)
class ModeSpecificity:
    """Per-mode rejection rates and group-vs-stage projection diagnostic."""

    mode: str
    rejection_rates: dict[str, float]
    group_in_stage_fraction: float
    n_replicates: int
    integration_method: str = "concat"


@dataclass(frozen=True)
class ShapeNullDiagnostic:
    """Observed Procrustes distance vs its RRPP permutation null, per mode.

    Splits the ``shape`` rejection into its two ingredients so the saturation
    can be attributed: ``observed_mean`` is the average observed group-vs-group
    Procrustes distance, and the ``null_*`` summaries describe the spread of the
    permutation null. An anti-conservative test shows a collapsing null (small
    ``null_spread_mean``) rather than an extreme observed distance.
    """

    mode: str
    integration_method: str
    standardize: bool
    observed_mean: float
    null_q025_mean: float
    null_median_mean: float
    null_q975_mean: float
    null_spread_mean: float
    rejection_rate: float
    n_replicates: int


def _group_in_stage_fraction(dataset, params: SimulationEvaluationParams) -> float:
    """Fraction of the group mean-difference energy lying in the stage subspace.

    Projects the concatenated group mean-difference vector onto the span of the
    stage LS-mean directions; ~1 means the injected group signal is disease
    (stage) relevant, ~0 means it is orthogonal to it.
    """

    latent = integrate_semisynthetic_dataset(dataset, params)
    Y = latent.matrix.to_numpy(dtype=float)
    meta = dataset.metadata
    groups = meta[params.group_col].to_numpy()
    stages = meta[params.stage_col].to_numpy()
    g_labels = sorted(set(groups))
    group_diff = Y[groups == g_labels[1]].mean(0) - Y[groups == g_labels[0]].mean(0)
    norm = np.linalg.norm(group_diff)
    if norm < 1e-12:
        return 0.0

    # Stage subspace: centered per-stage means (disease-relevant directions).
    stage_means = np.vstack([Y[stages == s].mean(0) for s in sorted(set(stages))])
    stage_means = stage_means - stage_means.mean(0, keepdims=True)
    basis, _ = np.linalg.qr(stage_means.T)
    projected = basis @ (basis.T @ group_diff)
    return float(np.linalg.norm(projected) / norm)


def evaluate_mode_specificity(
    mode: TrajectoryMode,
    *,
    n_replicates: int = 10,
    n_samples: int = 180,
    n_stages: int = 4,
    effect_size: float = 1.0,
    p_dmp: float = 0.2,
    shape_kind: str = "relocate",
    magnitude_kind: str = "all",
    permutations: int = 99,
    alpha: float = 0.05,
    n_jobs: int | None = -1,
    base_seed: int = 0,
    reference: IntersimReference | None = None,
    integration_method: str = "concat",
    integration_params: dict[str, object] | None = None,
) -> ModeSpecificity:
    """Run replicates for one mode and report per-statistic rejection rates.

    ``integration_method`` selects the latent space the trajectory is measured
    in — the ``concat`` baseline (default), ``snf``, or the ``pls`` production
    latent space — and ``integration_params`` is forwarded to it (e.g. the PLS
    cross-validation knobs). The same selection drives both the RRPP
    rejection-rate evaluation and the group-in-stage projection.
    """

    ref = reference if reference is not None else load_reference()
    int_params = dict(integration_params or {})
    eval_params = SimulationEvaluationParams(
        permutations=permutations,
        n_jobs=n_jobs,
        integration_method=integration_method,  # type: ignore[arg-type]
        integration_params=int_params,
    )
    rejections = {stat: 0 for stat in STATISTICS}
    available = {stat: 0 for stat in STATISTICS}
    gis_values: list[float] = []

    for rep in range(n_replicates):
        params = SemiSyntheticTrajectoryParams(
            seed=base_seed + rep,
            trajectory_mode=mode,
            n_samples=n_samples,
            n_stages=n_stages,
            group_effect_size=effect_size,
            p_dmp=p_dmp,
            shape_kind=shape_kind,  # type: ignore[arg-type]
            magnitude_kind=magnitude_kind,  # type: ignore[arg-type]
        )
        dataset = generate_semisynthetic_trajectory(params, reference=ref)
        result = evaluate_semisynthetic_trajectory(
            dataset,
            SimulationEvaluationParams(
                permutations=permutations,
                n_jobs=n_jobs,
                seed=base_seed + rep,
                integration_method=integration_method,  # type: ignore[arg-type]
                integration_params=int_params,
            ),
        )
        for stat in STATISTICS:
            p = result.p_values.get(stat)
            if p is not None and np.isfinite(p):
                available[stat] += 1
                if p < alpha:
                    rejections[stat] += 1
        gis_values.append(_group_in_stage_fraction(dataset, eval_params))

    rates = {
        stat: (rejections[stat] / available[stat] if available[stat] else float("nan"))
        for stat in STATISTICS
    }
    return ModeSpecificity(
        mode=mode,
        rejection_rates=rates,
        group_in_stage_fraction=float(np.mean(gis_values)),
        n_replicates=n_replicates,
        integration_method=integration_method,
    )


def characterize_two_stage(
    *,
    modes: tuple[str, ...] = SHAPE_FREE_MODES,
    n_replicates: int = 10,
    n_samples: int = 180,
    effect_size: float = 1.0,
    p_dmp: float = 0.2,
    permutations: int = 99,
    alpha: float = 0.05,
    n_jobs: int | None = -1,
    base_seed: int = 0,
    reference: IntersimReference | None = None,
    integration_method: str = "concat",
    integration_params: dict[str, object] | None = None,
) -> dict[str, ModeSpecificity]:
    """Run the shape-free (``n_stages=2``) isolation pass for each mode.

    With two stages the trajectory is a single step, so Procrustes ``shape`` is
    degenerate (the evaluation reports it as ``nan`` and it never counts). This
    isolates ``magnitude``→``delta`` and ``orientation``→``angle`` with shape out
    of the picture — the cleanest test of whether the constructions are specific
    or whether the 3/4-stage cross-talk was shape contaminating them.

    ``integration_method``/``integration_params`` select the latent space, as in
    :func:`evaluate_mode_specificity`.
    """

    ref = reference if reference is not None else load_reference()
    return {
        mode: evaluate_mode_specificity(
            mode,  # type: ignore[arg-type]
            n_replicates=n_replicates,
            n_samples=n_samples,
            n_stages=2,
            effect_size=effect_size,
            p_dmp=p_dmp,
            permutations=permutations,
            alpha=alpha,
            n_jobs=n_jobs,
            base_seed=base_seed,
            reference=ref,
            integration_method=integration_method,
            integration_params=integration_params,
        )
        for mode in modes
    }


def evaluate_shape_null(
    mode: TrajectoryMode,
    *,
    integration_method: str = "concat",
    standardize: bool = True,
    n_replicates: int = 10,
    n_samples: int = 180,
    n_stages: int = 4,
    effect_size: float = 1.0,
    p_dmp: float = 0.2,
    shape_kind: str = "relocate",
    magnitude_kind: str = "all",
    permutations: int = 99,
    alpha: float = 0.05,
    n_jobs: int | None = -1,
    base_seed: int = 0,
    reference: IntersimReference | None = None,
) -> ShapeNullDiagnostic:
    """Split the ``shape`` rejection into observed distance vs permutation null.

    For each replicate, records the observed group-vs-group Procrustes distance
    and the quantiles/spread of its RRPP permutation null (via
    ``include_null_distributions``). ``integration_method``/``standardize`` select
    raw concat (``standardize=False``), concat-standardize (the default), or SNF
    — the probe for whether per-feature standardization is what breaks
    Procrustes scale-invariance.
    """

    ref = reference if reference is not None else load_reference()
    integration_params: dict[str, object] = (
        {} if integration_method == "snf" else {"standardize": standardize}
    )

    observed: list[float] = []
    q025: list[float] = []
    medians: list[float] = []
    q975: list[float] = []
    spreads: list[float] = []
    rejections = 0
    available = 0

    for rep in range(n_replicates):
        params = SemiSyntheticTrajectoryParams(
            seed=base_seed + rep,
            trajectory_mode=mode,
            n_samples=n_samples,
            n_stages=n_stages,
            group_effect_size=effect_size,
            p_dmp=p_dmp,
            shape_kind=shape_kind,  # type: ignore[arg-type]
            magnitude_kind=magnitude_kind,  # type: ignore[arg-type]
        )
        dataset = generate_semisynthetic_trajectory(params, reference=ref)
        result = evaluate_semisynthetic_trajectory(
            dataset,
            SimulationEvaluationParams(
                integration_method=integration_method,  # type: ignore[arg-type]
                integration_params=integration_params,
                permutations=permutations,
                n_jobs=n_jobs,
                seed=base_seed + rep,
                include_null_distributions=True,
            ),
        )
        obs = result.pair_statistics.get("shape")
        null = (result.null_distributions or {}).get("shape")
        if obs is None or not np.isfinite(obs) or not null:
            continue
        null_arr = np.asarray(null, dtype=float)
        observed.append(float(obs))
        q025.append(float(np.quantile(null_arr, 0.025)))
        medians.append(float(np.median(null_arr)))
        q975.append(float(np.quantile(null_arr, 0.975)))
        spreads.append(float(null_arr.std()))
        available += 1
        p = result.p_values.get("shape")
        if p is not None and np.isfinite(p) and p < alpha:
            rejections += 1

    def _mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    return ShapeNullDiagnostic(
        mode=mode,
        integration_method=integration_method,
        standardize=standardize,
        observed_mean=_mean(observed),
        null_q025_mean=_mean(q025),
        null_median_mean=_mean(medians),
        null_q975_mean=_mean(q975),
        null_spread_mean=_mean(spreads),
        rejection_rate=(rejections / available if available else float("nan")),
        n_replicates=available,
    )


def target_leads(report: ModeSpecificity) -> bool:
    """Descriptive flag: does the mode's target statistic lead the response?

    Informational only — **not** a pass/fail gate. For non-null modes it reports
    whether the target statistic has the highest rejection rate and clears 0.5;
    for the negative controls (``none``/``translation``) it reports whether no
    statistic rejects strongly. Cross-talk is expected; a ``False`` here is a
    finding to characterize, not a failure to fix.
    """

    rates = report.rejection_rates
    target = TARGET_STATISTIC.get(report.mode)
    if target is None:  # none / translation -> negative controls
        return max(rates.values()) <= 0.5
    others = [v for k, v in rates.items() if k != target]
    return rates[target] >= max(others) and rates[target] >= 0.5
