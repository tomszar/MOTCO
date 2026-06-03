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


@dataclass(frozen=True)
class ModeSpecificity:
    """Per-mode rejection rates and group-vs-stage projection diagnostic."""

    mode: str
    rejection_rates: dict[str, float]
    group_in_stage_fraction: float
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
    permutations: int = 99,
    alpha: float = 0.05,
    n_jobs: int | None = -1,
    base_seed: int = 0,
    reference: IntersimReference | None = None,
) -> ModeSpecificity:
    """Run replicates for one mode and report per-statistic rejection rates."""

    ref = reference if reference is not None else load_reference()
    eval_params = SimulationEvaluationParams(permutations=permutations, n_jobs=n_jobs)
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
        )
        dataset = generate_semisynthetic_trajectory(params, reference=ref)
        result = evaluate_semisynthetic_trajectory(
            dataset, SimulationEvaluationParams(permutations=permutations, n_jobs=n_jobs, seed=base_seed + rep)
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
