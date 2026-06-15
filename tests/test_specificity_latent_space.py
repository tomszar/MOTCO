from __future__ import annotations

from motco.simulations.specificity import (
    STATISTICS,
    characterize_two_stage,
    evaluate_mode_specificity,
)

# Tiny CV knobs keep the PLS double-CV cheap for the fast suite.
_PLS_CV = {"cv2_splits": 2, "cv1_splits": 2, "n_repeats": 1, "max_components": 4}
_COMMON = dict(n_replicates=1, n_samples=80, n_stages=2, permutations=5, n_jobs=1, base_seed=0)


def test_mode_specificity_runs_through_pls_latent_space() -> None:
    result = evaluate_mode_specificity(
        "magnitude",
        integration_method="pls",
        integration_params=_PLS_CV,
        **_COMMON,
    )

    assert result.integration_method == "pls"
    # Rejection rates are reported for every statistic (shape is nan at 2 stages).
    assert set(result.rejection_rates) == set(STATISTICS)
    assert 0.0 <= result.group_in_stage_fraction <= 1.0


def test_mode_specificity_defaults_to_concat_baseline() -> None:
    result = evaluate_mode_specificity("magnitude", **_COMMON)

    assert result.integration_method == "concat"


def test_characterize_two_stage_forwards_latent_space() -> None:
    results = characterize_two_stage(
        modes=("magnitude",),
        integration_method="pls",
        integration_params=_PLS_CV,
        n_replicates=1,
        n_samples=80,
        permutations=5,
        n_jobs=1,
        base_seed=0,
    )

    assert results["magnitude"].integration_method == "pls"
