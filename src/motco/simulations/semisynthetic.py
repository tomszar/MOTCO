"""Semi-synthetic trajectory datasets from the numpy generator.

Trajectory differences between two groups are defined as **feature-set surgery
on the methylation differential indicators only** — gene-expression and protein
differential indicators are always *re-derived* from the (group-specific)
methylation indicators through the cached CpG→gene→protein incidence maps. This
honours the biological cascade (methylation drives expression drives protein)
and keeps the datasets realistic rather than tailored to MOTCO: we manipulate
only the original methylation features and never the latent space.

Group A inherits a random baseline trajectory (independent per-stage methylation
indicators, intentionally *not* forced continuous). Group B is a deterministic
transform of A's **methylation** indicators:

- ``none``        -- identical indicators (null).
- ``translation`` -- A's stage-changing sites unchanged, plus an extra set ``U``
  of methylation sites (disjoint from the stage-changing sites) differential at
  *every* B stage and at none of A's. A constant group offset → moves only the
  (untested) group main effect, not size/orientation/shape.
- ``magnitude``   -- same indicators, scaled methylation effect
  ``δ_methyl_B = (1 + e)·δ_methyl`` → uniformly scales every methylation step.
- ``orientation`` -- relocate a fraction ``e`` of the stage-changing sites to
  different CpGs, the **same relocation at every stage** → the per-stage pattern
  runs along different feature axes (a rotation).
- ``shape``       -- perturb a **single interior stage** (≥3 stages): relocate a
  fraction ``e`` of that stage's sites (``shape_kind='relocate'``) or scale that
  stage's methylation effect (``shape_kind='magnitude'``) → bends one vertex.

``group_effect_size`` (``e``) is the unified knob (``e = 0`` is null for every
mode). Cross-talk between statistics (e.g. magnitude bending shape via the
methylation ``rev.logit`` nonlinearity) is *expected and reported*, not
engineered away — how well MOTCO separates the modes is an open question the
study characterizes. Generation runs entirely on the numpy generator and cached
reference data — no R at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

from motco.simulations.generator import (
    GeneratedOmics,
    bernoulli_indicators,
    derive_coupled_indicators,
    generate_omics,
)
from motco.simulations.reference import IntersimReference, load_reference

OmicsLayer = Literal["methylation", "expression", "proteomics"]
TrajectoryMode = Literal["none", "translation", "magnitude", "orientation", "shape"]
ShapeKind = Literal["relocate", "magnitude"]

_OMICS_LAYERS: tuple[OmicsLayer, ...] = ("methylation", "expression", "proteomics")
_MODES = frozenset({"none", "translation", "magnitude", "orientation", "shape"})
_SHAPE_KINDS = frozenset({"relocate", "magnitude"})


class SemiSyntheticTrajectoryError(ValueError):
    """Raised when semi-synthetic trajectory generation parameters are invalid."""


@dataclass(frozen=True)
class SemiSyntheticTrajectoryParams:
    """Parameters for semi-synthetic trajectory generation.

    ``group_ratio`` is the proportion assigned to the first label in
    ``group_labels`` within every stage. ``group_effect_size`` is the unified
    effect knob described in the module docstring (``0`` is null for all modes).
    ``p_dmp`` is the per-stage probability that a methylation feature is
    differential (InterSIM's ``p.DMP``); expression/protein indicators are
    derived from it via the cross-omic maps. ``delta_*`` are the per-omic
    mean-shift sizes (InterSIM's ``delta.*``). ``shape_kind`` selects the
    single-interior-stage perturbation used by ``shape``.
    """

    seed: int
    trajectory_mode: TrajectoryMode = "none"
    n_samples: int = 600
    n_stages: int = 3
    group_effect_size: float = 0.0
    group_ratio: float = 0.5
    group_labels: tuple[str, str] = ("A", "B")
    p_dmp: float = 0.2
    delta_methyl: float = 2.0
    delta_expr: float = 2.0
    delta_protein: float = 2.0
    shape_kind: ShapeKind = "relocate"
    stage_sample_prop: tuple[float, ...] | None = None


@dataclass(frozen=True)
class SemiSyntheticTrajectoryDataset:
    """MOTCO-ready semi-synthetic trajectory dataset."""

    methylation: pd.DataFrame
    expression: pd.DataFrame
    proteomics: pd.DataFrame
    metadata: pd.DataFrame
    truth: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _GroupIndicators:
    methyl: np.ndarray
    expr: np.ndarray
    protein: np.ndarray


def generate_semisynthetic_trajectory(
    params: SemiSyntheticTrajectoryParams,
    *,
    reference: IntersimReference | None = None,
) -> SemiSyntheticTrajectoryDataset:
    """Generate a semi-synthetic trajectory dataset using the numpy generator."""

    _validate_params(params)
    ref = reference if reference is not None else load_reference()
    rng = np.random.default_rng(params.seed)

    stage_sizes = _stage_sizes(params)
    group_a_sizes, group_b_sizes = _group_stage_sizes(stage_sizes, params)

    methyl_a = _baseline_methyl(rng, ref, params)
    methyl_b, deltas_b, transform_meta = _transform_group_b(rng, ref, params, methyl_a)
    deltas_a = (params.delta_methyl, params.delta_expr, params.delta_protein)

    indicators_a = _derive_group(methyl_a, ref)
    indicators_b = _derive_group(methyl_b, ref)

    gen_a = _generate_group(rng, ref, indicators_a, deltas_a, group_a_sizes)
    gen_b = _generate_group(rng, ref, indicators_b, deltas_b, group_b_sizes)

    methylation, expression, proteomics, metadata = _assemble(
        ref, params, gen_a, gen_b, group_a_sizes, group_b_sizes
    )
    truth = _build_truth(params, indicators_a, indicators_b, deltas_a, deltas_b, transform_meta)

    return SemiSyntheticTrajectoryDataset(
        methylation=methylation,
        expression=expression,
        proteomics=proteomics,
        metadata=metadata,
        truth=truth,
    )


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def _validate_params(params: SemiSyntheticTrajectoryParams) -> None:
    if params.trajectory_mode not in _MODES:
        raise SemiSyntheticTrajectoryError(f"Unknown trajectory_mode: {params.trajectory_mode}")
    if params.shape_kind not in _SHAPE_KINDS:
        raise SemiSyntheticTrajectoryError(f"Unknown shape_kind: {params.shape_kind}")
    if len(params.group_labels) != 2 or params.group_labels[0] == params.group_labels[1]:
        raise SemiSyntheticTrajectoryError("group_labels must contain two distinct labels.")
    if not (0 < params.group_ratio < 1):
        raise SemiSyntheticTrajectoryError("group_ratio must be between 0 and 1.")
    if params.group_effect_size < 0:
        raise SemiSyntheticTrajectoryError("group_effect_size must be non-negative.")
    if not (0 <= params.p_dmp <= 1):
        raise SemiSyntheticTrajectoryError("p_dmp must be between 0 and 1.")
    if params.n_stages < 2:
        raise SemiSyntheticTrajectoryError("n_stages must be at least 2.")
    if params.trajectory_mode == "shape" and params.n_stages < 3:
        raise SemiSyntheticTrajectoryError("trajectory_mode='shape' requires at least three stages.")
    for name in ("delta_methyl", "delta_expr", "delta_protein"):
        if getattr(params, name) < 0:
            raise SemiSyntheticTrajectoryError(f"{name} must be non-negative.")
    if params.stage_sample_prop is not None:
        if len(params.stage_sample_prop) != params.n_stages:
            raise SemiSyntheticTrajectoryError("stage_sample_prop must have one entry per stage.")
        if abs(sum(params.stage_sample_prop) - 1.0) > 1e-6:
            raise SemiSyntheticTrajectoryError("stage_sample_prop must sum to 1.")


# --------------------------------------------------------------------------- #
# Sizing
# --------------------------------------------------------------------------- #


def _stage_sizes(params: SemiSyntheticTrajectoryParams) -> list[int]:
    k = params.n_stages
    prop = params.stage_sample_prop or tuple([1.0 / k] * k)
    sizes = [int(round(params.n_samples * prop[i])) for i in range(k - 1)]
    sizes.append(params.n_samples - sum(sizes))
    if any(n < 2 for n in sizes):
        raise SemiSyntheticTrajectoryError(
            "Each stage needs at least 2 samples (one per group); increase n_samples."
        )
    return sizes


def _group_stage_sizes(
    stage_sizes: list[int], params: SemiSyntheticTrajectoryParams
) -> tuple[list[int], list[int]]:
    a_sizes, b_sizes = [], []
    for n in stage_sizes:
        a = int(round(n * params.group_ratio))
        a = min(max(a, 1), n - 1)
        a_sizes.append(a)
        b_sizes.append(n - a)
    return a_sizes, b_sizes


# --------------------------------------------------------------------------- #
# Methylation baseline and the group-B transform (methylation only)
# --------------------------------------------------------------------------- #


def _baseline_methyl(
    rng: np.random.Generator,
    ref: IntersimReference,
    params: SemiSyntheticTrajectoryParams,
) -> np.ndarray:
    """Group A's per-stage methylation differential indicators (binary)."""

    return bernoulli_indicators(rng, ref.n_cpg, params.n_stages, params.p_dmp)


def _derive_group(methyl: np.ndarray, ref: IntersimReference) -> _GroupIndicators:
    """Derive expression and protein indicators from methylation via the cascade."""

    expr, protein = derive_coupled_indicators((methyl > 0).astype(float), ref)
    return _GroupIndicators(methyl=methyl, expr=expr, protein=protein)


def _transform_group_b(
    rng: np.random.Generator,
    ref: IntersimReference,
    params: SemiSyntheticTrajectoryParams,
    methyl_a: np.ndarray,
) -> tuple[np.ndarray, tuple[float, float, float], dict[str, Any]]:
    """Return group B's methylation indicators, per-omic deltas, and truth notes."""

    deltas_a = (params.delta_methyl, params.delta_expr, params.delta_protein)
    e = params.group_effect_size
    mode = params.trajectory_mode

    if mode == "none" or e == 0:
        return methyl_a.copy(), deltas_a, {}

    if mode == "translation":
        return _translation_methyl(rng, ref, params, methyl_a)

    if mode == "magnitude":
        scaled = (float((1.0 + e) * params.delta_methyl), params.delta_expr, params.delta_protein)
        return methyl_a.copy(), scaled, {"delta_methyl_scale": 1.0 + e}

    if mode == "orientation":
        return _orientation_methyl(rng, methyl_a, e, params)

    if mode == "shape":
        return _shape_methyl(rng, methyl_a, e, params)

    raise SemiSyntheticTrajectoryError(f"Unknown trajectory_mode: {mode}")


def _translation_methyl(
    rng: np.random.Generator,
    ref: IntersimReference,
    params: SemiSyntheticTrajectoryParams,
    methyl_a: np.ndarray,
) -> tuple[np.ndarray, tuple[float, float, float], dict[str, Any]]:
    """Add an extra constant set U of DMPs on an independent gene program.

    For a clean location-only control, U's CpGs must regulate genes that the
    stage trajectory does *not* touch — otherwise the CpG→gene OR-derivation
    would let U saturate (flatten) stage-varying genes in group B, deforming the
    derived trajectory. So U is drawn from stage-inactive CpGs whose mapped gene
    is also absent from the stage program.
    """

    methyl_b = methyl_a.copy()
    stage_active = methyl_a.sum(1) > 0
    used_genes = ref.incidence_cpg_gene[stage_active].sum(0) > 0  # genes in the stage program
    cpg_gene = ref.incidence_cpg_gene.argmax(1)  # each CpG's mapped gene
    fresh = (~stage_active) & (~used_genes[cpg_gene])  # stage-inactive CpGs on fresh genes
    candidates = np.where(fresh)[0]
    n_extra = int(round(params.group_effect_size * params.p_dmp * ref.n_cpg))
    n_extra = min(n_extra, len(candidates))
    if n_extra >= 1:
        u = rng.choice(candidates, size=n_extra, replace=False)
        methyl_b[u, :] = 1.0  # differential at every B stage, none of A's
    deltas = (params.delta_methyl, params.delta_expr, params.delta_protein)
    return methyl_b, deltas, {"translation_set_size": int(n_extra)}


def _relocate_rows(
    rng: np.random.Generator, src_pool: np.ndarray, dst_pool: np.ndarray, fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    """Pick ``fraction`` of ``src_pool`` and an equal number from ``dst_pool``."""

    k = int(round(min(max(fraction, 0.0), 1.0) * len(src_pool)))
    k = min(k, len(dst_pool))
    if k < 1:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    src = rng.choice(src_pool, size=k, replace=False)
    dst = rng.choice(dst_pool, size=k, replace=False)
    return src, dst


def _orientation_methyl(
    rng: np.random.Generator,
    methyl_a: np.ndarray,
    e: float,
    params: SemiSyntheticTrajectoryParams,
) -> tuple[np.ndarray, tuple[float, float, float], dict[str, Any]]:
    """Relocate a fraction of stage-changing sites, the same relocation at every stage."""

    active = np.where(methyl_a.sum(1) > 0)[0]
    inactive = np.where(methyl_a.sum(1) == 0)[0]
    src, dst = _relocate_rows(rng, active, inactive, e)
    methyl_b = methyl_a.copy()
    if src.size:
        methyl_b[dst, :] = methyl_a[src, :]  # move whole rows → same relocation per stage
        methyl_b[src, :] = 0.0
    deltas = (params.delta_methyl, params.delta_expr, params.delta_protein)
    return methyl_b, deltas, {"orientation_relocated": int(src.size)}


def _shape_methyl(
    rng: np.random.Generator,
    methyl_a: np.ndarray,
    e: float,
    params: SemiSyntheticTrajectoryParams,
) -> tuple[np.ndarray, tuple[float, float, float], dict[str, Any]]:
    """Perturb a single interior stage: relocate its sites or scale its effect."""

    methyl_b = methyl_a.astype(float).copy()
    interior = list(range(1, params.n_stages - 1))
    stage = interior[len(interior) // 2]  # the single interior stage to perturb
    meta: dict[str, Any] = {"shape_stage": stage, "shape_kind": params.shape_kind}

    if params.shape_kind == "magnitude":
        methyl_b[:, stage] = methyl_a[:, stage] * (1.0 + e)
        meta["shape_scale"] = 1.0 + e
    else:  # relocate this stage's sites to globally-inactive CpGs → bends the vertex
        active_here = np.where(methyl_a[:, stage] > 0)[0]
        global_inactive = np.where(methyl_a.sum(1) == 0)[0]
        src, dst = _relocate_rows(rng, active_here, global_inactive, e)
        if src.size:
            methyl_b[dst, stage] = methyl_a[src, stage]
            methyl_b[src, stage] = 0.0
        meta["shape_relocated"] = int(src.size)
    deltas = (params.delta_methyl, params.delta_expr, params.delta_protein)
    return methyl_b, deltas, meta


# --------------------------------------------------------------------------- #
# Generation + assembly
# --------------------------------------------------------------------------- #


def _generate_group(
    rng: np.random.Generator,
    ref: IntersimReference,
    indicators: _GroupIndicators,
    deltas: tuple[float, float, float],
    cell_sizes: list[int],
) -> GeneratedOmics:
    return generate_omics(
        cell_sizes=cell_sizes,
        indicators_methyl=indicators.methyl,
        indicators_expr=indicators.expr,
        indicators_protein=indicators.protein,
        delta_methyl=deltas[0],
        delta_expr=deltas[1],
        delta_protein=deltas[2],
        rng=rng,
        reference=ref,
    )


def _assemble(
    ref: IntersimReference,
    params: SemiSyntheticTrajectoryParams,
    gen_a: GeneratedOmics,
    gen_b: GeneratedOmics,
    a_sizes: list[int],
    b_sizes: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    label_a, label_b = params.group_labels
    groups = [label_a] * sum(a_sizes) + [label_b] * sum(b_sizes)
    stages = np.concatenate([gen_a.cell_ids, gen_b.cell_ids]).astype(int)
    sample_ids = [f"sample{i}" for i in range(len(groups))]

    metadata = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "group": groups,
            "stage": stages,
            "cluster": stages,
        },
        index=sample_ids,
    )

    methylation = pd.DataFrame(
        np.vstack([gen_a.methylation, gen_b.methylation]),
        index=sample_ids,
        columns=ref.cpg_names,
    )
    expression = pd.DataFrame(
        np.vstack([gen_a.expression, gen_b.expression]),
        index=sample_ids,
        columns=ref.gene_names,
    )
    proteomics = pd.DataFrame(
        np.vstack([gen_a.proteomics, gen_b.proteomics]),
        index=sample_ids,
        columns=ref.protein_names,
    )
    return methylation, expression, proteomics, metadata


def _build_truth(
    params: SemiSyntheticTrajectoryParams,
    indicators_a: _GroupIndicators,
    indicators_b: _GroupIndicators,
    deltas_a: tuple[float, float, float],
    deltas_b: tuple[float, float, float],
    transform_meta: dict[str, Any],
) -> dict[str, Any]:
    def counts(ind: _GroupIndicators) -> dict[str, list[int]]:
        return {
            "methylation": (ind.methyl != 0).sum(0).astype(int).tolist(),
            "expression": (ind.expr != 0).sum(0).astype(int).tolist(),
            "proteomics": (ind.protein != 0).sum(0).astype(int).tolist(),
        }

    label_a, label_b = params.group_labels
    return {
        "trajectory_mode": params.trajectory_mode,
        "group_effect_size": params.group_effect_size,
        "group_labels": [label_a, label_b],
        "group_ratio": params.group_ratio,
        "n_stages": params.n_stages,
        "p_dmp": params.p_dmp,
        "shape_kind": params.shape_kind,
        "seed": params.seed,
        "stage_assumption": "clusters-as-stages",
        "deltas": {label_a: list(deltas_a), label_b: list(deltas_b)},
        "indicator_counts": {label_a: counts(indicators_a), label_b: counts(indicators_b)},
        "indicators": {
            label_a: {
                "methylation": indicators_a.methyl,
                "expression": indicators_a.expr,
                "proteomics": indicators_a.protein,
            },
            label_b: {
                "methylation": indicators_b.methyl,
                "expression": indicators_b.expr,
                "proteomics": indicators_b.protein,
            },
        },
        "transform": transform_meta,
    }


# --------------------------------------------------------------------------- #
# Convenience helpers consumed downstream
# --------------------------------------------------------------------------- #


def affected_omics_layers() -> tuple[OmicsLayer, ...]:
    """Canonical omic-layer order."""

    return _OMICS_LAYERS


def list_trajectory_modes() -> Sequence[TrajectoryMode]:
    """All supported trajectory modes."""

    return ("none", "translation", "magnitude", "orientation", "shape")
