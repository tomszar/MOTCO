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

from .pls import plsda_doubleCV  # noqa: F401
from .snf import SNF, get_affinity_matrix, get_spectral  # noqa: F401
from .sd import RRPP, estimate_difference  # noqa: F401

__all__ = [
    "plsda_doubleCV",
    "SNF",
    "get_affinity_matrix",
    "get_spectral",
    "RRPP",
    "estimate_difference",
]
