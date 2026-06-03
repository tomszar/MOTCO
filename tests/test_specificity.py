from __future__ import annotations

import pytest

from motco.simulations.reference import load_reference
from motco.simulations.specificity import (
    STATISTICS,
    ModeSpecificity,
    evaluate_mode_specificity,
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
