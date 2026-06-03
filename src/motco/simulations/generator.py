"""numpy-native multi-omic generator reproducing InterSIM's mean-shift model.

This replaces the per-replicate R subprocess. InterSIM's generative model
(read from ``body(InterSIM)``) is, per omic and per cluster ``i``::

    effect_i = base + delta * v_i          # v_i in {0,1}^p is the differential indicator
    X_i      ~ MVN(effect_i, Sigma)        # methylation then passes through rev.logit

with cross-omic coupling: differential genes are derived from differential CpGs
and differential proteins from differential genes, and expression/protein means
blend a per-feature cross-omic correlation term. All reference objects come from
the cached :class:`~motco.simulations.reference.IntersimReference`; no R is used
at runtime.

The trajectory layer treats each *cell* (a group x stage combination) as one
"cluster" column, so the same core handles InterSIM-style clustering and the
feature-surgery trajectory modes uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from motco.simulations.reference import IntersimReference, load_reference

#: Omic layer names in canonical order.
OMIC_LAYERS = ("methylation", "expression", "proteomics")


class GeneratorError(ValueError):
    """Raised for invalid generator inputs."""


@dataclass(frozen=True)
class GeneratedOmics:
    """Aligned omics matrices plus the indicator truth used to generate them.

    ``methylation`` / ``expression`` / ``proteomics`` are ``(n_samples, n_feat)``
    matrices sharing the row order given by ``cell_ids`` (the index of the cell
    each sample was drawn from). ``indicators_*`` are ``(n_feat, n_cell)`` {0,1}
    matrices recording the differential-feature indicator for each cell.
    """

    methylation: np.ndarray
    expression: np.ndarray
    proteomics: np.ndarray
    cell_ids: np.ndarray
    indicators_methyl: np.ndarray
    indicators_expr: np.ndarray
    indicators_protein: np.ndarray


def rev_logit(x: np.ndarray) -> np.ndarray:
    """InterSIM's inverse-logit: ``1 / (1 + exp(-x))``."""

    return 1.0 / (1.0 + np.exp(-x))


def bernoulli_indicators(
    rng: np.random.Generator, n_feat: int, n_cell: int, p: float
) -> np.ndarray:
    """Independent Bernoulli(``p``) differential indicator per feature and cell."""

    return (rng.random((n_feat, n_cell)) < p).astype(float)


def derive_coupled_indicators(
    indicators_methyl: np.ndarray, reference: IntersimReference
) -> tuple[np.ndarray, np.ndarray]:
    """Derive expression and protein indicators from methylation indicators.

    Mirrors InterSIM's default coupling: a gene is differential when any of its
    mapped CpGs is, and a protein is differential when its mapped gene is. Uses
    the cached incidence matrices, so the result matches InterSIM's
    ``CpG.gene.map.for.DEG`` / ``protein.gene.map.for.DEP`` derivation.
    """

    expr = (reference.incidence_cpg_gene.T @ indicators_methyl > 0).astype(float)
    protein = (reference.incidence_gene_protein.T @ expr > 0).astype(float)
    return expr, protein


def _sample_cells(
    rng: np.random.Generator,
    base: np.ndarray,
    delta: float,
    indicators: np.ndarray,
    cov: np.ndarray,
    cell_sizes: list[int],
    base_shift: np.ndarray | None = None,
) -> np.ndarray:
    """Stack MVN draws for each cell with mean ``base + base_shift + delta * indicator``.

    ``base_shift`` is a constant per-feature offset added to every cell's mean
    (used for the location-only ``translation`` mode, applied in the pre-logit
    M-value space for methylation).
    """

    shifted_base = base if base_shift is None else base + base_shift
    blocks = []
    for i, n in enumerate(cell_sizes):
        if n == 0:
            continue
        mean = shifted_base + delta * indicators[:, i]
        blocks.append(rng.multivariate_normal(mean, cov, size=n, method="svd"))
    return np.vstack(blocks)


def _resolve_cov(
    sigma: str | np.ndarray | None, default: np.ndarray
) -> np.ndarray:
    if sigma is None:
        return default
    if isinstance(sigma, str):
        if sigma == "indep":
            return np.diag(np.diag(default))
        raise GeneratorError(f"Unknown covariance option: {sigma!r}")
    return np.asarray(sigma, dtype=float)


def generate_omics(
    *,
    cell_sizes: list[int],
    indicators_methyl: np.ndarray,
    indicators_expr: np.ndarray,
    indicators_protein: np.ndarray,
    delta_methyl: float = 2.0,
    delta_expr: float = 2.0,
    delta_protein: float = 2.0,
    rng: np.random.Generator,
    reference: IntersimReference | None = None,
    sigma_methyl: str | np.ndarray | None = None,
    sigma_expr: str | np.ndarray | None = None,
    sigma_protein: str | np.ndarray | None = None,
    base_shift_methyl: np.ndarray | None = None,
    base_shift_expr: np.ndarray | None = None,
    base_shift_protein: np.ndarray | None = None,
) -> GeneratedOmics:
    """Generate aligned omics matrices from explicit per-cell indicators.

    Each "cell" (column of the indicator matrices) is sampled as a cluster with
    mean ``base + delta * indicator`` and the reference covariance, exactly as
    InterSIM does. Methylation is shifted in M-value (logit) space and then
    passed through :func:`rev_logit`; expression and protein means carry the
    InterSIM cross-omic blend.
    """

    ref = reference if reference is not None else load_reference()
    n_cell = len(cell_sizes)
    for name, ind, n_feat in (
        ("methylation", indicators_methyl, ref.n_cpg),
        ("expression", indicators_expr, ref.n_gene),
        ("proteomics", indicators_protein, ref.n_protein),
    ):
        if ind.shape != (n_feat, n_cell):
            raise GeneratorError(
                f"{name} indicators must have shape ({n_feat}, {n_cell}); got {ind.shape}"
            )
    if any(n < 0 for n in cell_sizes):
        raise GeneratorError("cell_sizes must be non-negative")

    cov_M = _resolve_cov(sigma_methyl, ref.cov_M)
    cov_expr = _resolve_cov(sigma_expr, ref.cov_expr)
    cov_protein = _resolve_cov(sigma_protein, ref.cov_protein)

    # Methylation: additive shift in M-value space, then inverse-logit.
    methyl_M = _sample_cells(
        rng, ref.mean_M, delta_methyl, indicators_methyl, cov_M, cell_sizes, base_shift_methyl
    )
    methylation = rev_logit(methyl_M)

    # Expression: cross-omic blend with methylation; rho zeroed when no effect.
    rho_me = np.zeros_like(ref.rho_methyl_expr) if delta_expr == 0 else ref.rho_methyl_expr
    base_expr = rho_me * ref.methyl_gene_level_mean + np.sqrt(1 - rho_me**2) * ref.mean_expr
    expression = _sample_cells(
        rng, base_expr, delta_expr, indicators_expr, cov_expr, cell_sizes, base_shift_expr
    )

    # Protein: cross-omic blend with expression; rho zeroed when no effect.
    rho_ep = np.zeros_like(ref.rho_expr_protein) if delta_protein == 0 else ref.rho_expr_protein
    base_protein = rho_ep * ref.mean_expr_with_mapped_protein + np.sqrt(1 - rho_ep**2) * ref.mean_protein
    proteomics = _sample_cells(
        rng, base_protein, delta_protein, indicators_protein, cov_protein, cell_sizes, base_shift_protein
    )

    cell_ids = np.concatenate([np.full(n, i) for i, n in enumerate(cell_sizes) if n > 0])

    return GeneratedOmics(
        methylation=methylation,
        expression=expression,
        proteomics=proteomics,
        cell_ids=cell_ids,
        indicators_methyl=indicators_methyl,
        indicators_expr=indicators_expr,
        indicators_protein=indicators_protein,
    )
