"""Statistical utilities for MOTCO.

Modules
-------
pls
    Partial Least Squares (PLS-DA) utilities.
snf
    Similarity Network Fusion and spectral embedding.
sd
    Trajectory group differences (delta, angle, shape) and RRPP.
"""

from .pls import calculate_vips, plsda_doubleCV  # noqa: F401
from .sd import (  # noqa: F401
    RRPP,
    build_ls_means,
    center_matrix,
    estimate_betas,
    estimate_difference,
    get_model_matrix,
    get_observed_vectors,
    pair_difference,
)
from .snf import SNF, get_affinity_matrix, get_spectral  # noqa: F401

__all__ = [
    # pls
    "plsda_doubleCV",
    "calculate_vips",
    # sd
    "RRPP",
    "build_ls_means",
    "center_matrix",
    "estimate_betas",
    "estimate_difference",
    "get_model_matrix",
    "get_observed_vectors",
    "pair_difference",
    # snf
    "SNF",
    "get_affinity_matrix",
    "get_spectral",
]
