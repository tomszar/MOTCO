"""Statistical utilities for MOTCO.

Modules
-------
pls
    Partial Least Squares (PLS-DA) utilities.
snf
    Similarity Network Fusion and spectral embedding.
design
    Design matrix and LS-means construction.
trajectory
    Trajectory estimation and geometric metrics.
permutation
    RRPP permutation infrastructure.
attribution
    Feature attribution for two-group trajectories in a shared PLS space.
"""

from .attribution import (  # noqa: F401
    AggregateEffect,
    AttributionConfig,
    AttributionError,
    AttributionResult,
    BootstrapSummary,
    ComponentAttribution,
    FeatureEffect,
    InterpretationMetadata,
    OrientationAttributionResult,
    TransitionAttribution,
    analyze_orientation_attribution,
    attribution_frames,
    run_orientation_attribution,
    write_attribution_outputs,
)
from .design import build_ls_means, center_matrix, get_model_matrix  # noqa: F401
from .permutation import RRPP  # noqa: F401
from .pls import calculate_vips, fit_plsda_model, fit_plsda_transform, plsda_doubleCV  # noqa: F401
from .snf import SNF, get_affinity_matrix, get_spectral  # noqa: F401
from .trajectory import estimate_betas, estimate_difference, get_observed_vectors, pair_difference  # noqa: F401

__all__ = [
    # pls
    "plsda_doubleCV",
    "calculate_vips",
    "fit_plsda_model",
    "fit_plsda_transform",
    # design
    "build_ls_means",
    "center_matrix",
    "get_model_matrix",
    # trajectory
    "estimate_betas",
    "estimate_difference",
    "get_observed_vectors",
    "pair_difference",
    # attribution
    "AggregateEffect",
    "AttributionConfig",
    "AttributionError",
    "AttributionResult",
    "BootstrapSummary",
    "ComponentAttribution",
    "FeatureEffect",
    "InterpretationMetadata",
    "OrientationAttributionResult",
    "TransitionAttribution",
    "analyze_orientation_attribution",
    "attribution_frames",
    "run_orientation_attribution",
    "write_attribution_outputs",
    # permutation
    "RRPP",
    # snf
    "SNF",
    "get_affinity_matrix",
    "get_spectral",
]
