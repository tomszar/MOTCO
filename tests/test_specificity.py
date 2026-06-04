from __future__ import annotations

import pytest

from motco.simulations.reference import load_reference
from motco.simulations.specificity import (
    SHAPE_FREE_MODES,
    STATISTICS,
    ModeSpecificity,
    ShapeNullDiagnostic,
    characterize_two_stage,
    evaluate_mode_specificity,
    evaluate_shape_null,
    target_leads,
)


def _report(mode: str, delta: float, angle: float, shape: float) -> ModeSpecificity:
    return ModeSpecificity(
        mode=mode,
        rejection_rates={"delta": delta, "angle": angle, "shape": shape},
        group_in_stage_fraction=0.5,
        n_replicates=10,
    )


def test_target_leads_null_modes_require_low_rates():
    assert target_leads(_report("none", 0.0, 0.0, 0.0))
    assert target_leads(_report("translation", 0.1, 0.05, 0.0))
    # a null mode that rejects strongly is not specific
    assert not target_leads(_report("none", 0.9, 0.1, 0.1))


def test_target_leads_requires_target_statistic_to_lead():
    assert target_leads(_report("magnitude", 0.9, 0.2, 0.3))  # delta leads
    assert target_leads(_report("orientation", 0.1, 0.8, 0.2))  # angle leads
    assert target_leads(_report("shape", 0.2, 0.3, 0.7))  # shape leads
    # target present but not leading / too weak
    assert not target_leads(_report("magnitude", 0.3, 0.9, 0.2))
    assert not target_leads(_report("orientation", 0.1, 0.3, 0.1))


def test_shape_free_modes_excludes_shape():
    assert "shape" not in SHAPE_FREE_MODES
    assert set(SHAPE_FREE_MODES) == {"none", "translation", "magnitude", "orientation"}


@pytest.mark.slow
def test_evaluate_mode_specificity_runs_and_returns_valid_structure():
    report = evaluate_mode_specificity(
        "magnitude",
        n_replicates=2,
        n_samples=120,
        n_stages=4,
        effect_size=1.0,
        permutations=19,
        n_jobs=1,
        base_seed=0,
        reference=load_reference(),
    )
    assert report.mode == "magnitude"
    assert set(report.rejection_rates) == set(STATISTICS)
    assert all(0.0 <= r <= 1.0 for r in report.rejection_rates.values())
    assert 0.0 <= report.group_in_stage_fraction <= 1.0
    assert report.n_replicates == 2


@pytest.mark.slow
def test_characterize_two_stage_runs_shape_free_modes():
    reports = characterize_two_stage(
        modes=("none", "magnitude"),
        n_replicates=2,
        n_samples=120,
        effect_size=1.0,
        permutations=19,
        n_jobs=1,
        base_seed=0,
        reference=load_reference(),
    )
    assert set(reports) == {"none", "magnitude"}
    for report in reports.values():
        # shape is degenerate at n_stages=2 -> never rejects (nan rate)
        assert report.rejection_rates["shape"] != report.rejection_rates["shape"]  # nan
        assert 0.0 <= report.rejection_rates["delta"] <= 1.0
        assert 0.0 <= report.rejection_rates["angle"] <= 1.0


@pytest.mark.slow
def test_evaluate_shape_null_returns_observed_and_null_summaries():
    diag = evaluate_shape_null(
        "magnitude",
        integration_method="concat",
        standardize=True,
        n_replicates=2,
        n_samples=120,
        n_stages=4,
        effect_size=1.0,
        permutations=19,
        n_jobs=1,
        base_seed=0,
        reference=load_reference(),
    )
    assert isinstance(diag, ShapeNullDiagnostic)
    assert diag.mode == "magnitude"
    assert diag.integration_method == "concat"
    assert diag.standardize is True
    # observed Procrustes distance and the null quantiles are ordered and finite
    assert diag.null_q025_mean <= diag.null_median_mean <= diag.null_q975_mean
    assert diag.null_spread_mean >= 0.0
    assert 0.0 <= diag.rejection_rate <= 1.0
    assert diag.n_replicates == 2
