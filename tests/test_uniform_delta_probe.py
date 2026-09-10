"""The diagnostic-only uniform-delta magnitude construction.

The production ``magnitude_kind='all'`` scales methylation's delta alone, so the
concatenated trajectory rotates even though each omic block on its own is exactly
size-scaled. The uniform probe scales every omic's delta together. It must stay
unreachable from study configuration: adopting it is a method revision, not a
side effect.
"""

from __future__ import annotations

import json

import pytest

from motco.simulations.semisynthetic import (
    SemiSyntheticTrajectoryError,
    SemiSyntheticTrajectoryParams,
)
from motco.simulations.study.config import StudyConfigError, load_study_config


def test_uniform_probe_is_not_a_selectable_magnitude_kind():
    """Generation refuses it, so it cannot be reached through ``magnitude_kind``.

    The params dataclass itself does not validate; ``_validate_params`` runs at
    generation, which is where an unknown construction is rejected.
    """

    from motco.simulations.semisynthetic import generate_semisynthetic_trajectory

    params = SemiSyntheticTrajectoryParams(
        seed=0,
        trajectory_mode="magnitude",
        n_samples=60,
        n_stages=3,
        magnitude_kind="uniform_probe",  # type: ignore[arg-type]
    )
    with pytest.raises(SemiSyntheticTrajectoryError, match="Unknown magnitude_kind"):
        generate_semisynthetic_trajectory(params)


def test_study_config_refuses_the_probe_as_a_magnitude_kind(tmp_path):
    """Task 3.2 / design D4: a committed profile cannot acquire it silently."""

    def config(magnitude_kind: str) -> dict:
        return {
            "generator": {
                "seed": 1,
                "trajectory_mode": "magnitude",
                "magnitude_kind": magnitude_kind,
                "n_samples": 60,
                "n_stages": 3,
            },
            "evaluation": {"permutations": 9, "n_jobs": 1},
            "trajectory_modes": ["magnitude"],
            "effect_sizes": [0.0, 1.0],
            "n_replicates": 1,
        }

    # Control: an otherwise identical config with a real kind loads, so the
    # rejection below is about `magnitude_kind` and not a missing field.
    ok = tmp_path / "ok.json"
    ok.write_text(json.dumps(config("all")), encoding="utf-8")
    assert load_study_config(ok).generator.magnitude_kind == "all"

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(config("uniform_probe")), encoding="utf-8")
    with pytest.raises(StudyConfigError, match="generator.magnitude_kind"):
        load_study_config(bad)


def test_probe_keyword_rejects_non_magnitude_modes():
    from motco.simulations.semisynthetic import generate_semisynthetic_trajectory

    params = SemiSyntheticTrajectoryParams(
        seed=0,
        trajectory_mode="orientation",
        n_samples=60,
        n_stages=3,
        group_effect_size=1.0,
    )
    with pytest.raises(SemiSyntheticTrajectoryError, match="magnitude mode only"):
        generate_semisynthetic_trajectory(params, _probe_uniform_delta=True)


@pytest.mark.slow
def test_uniform_probe_is_size_pure_in_the_joint_space():
    """The finding: scaling every delta keeps the joint trajectory size-only.

    Production scaling rotates the concatenated trajectory (joint angle grows
    with the effect); the uniform probe leaves it at the floating-point floor.
    """

    from motco.simulations.specificity import compare_uniform_delta_construction

    rows = compare_uniform_delta_construction(
        effect_sizes=(0.0, 1.0),
        n_samples=180,
        n_stages=3,
        p_dmp=0.1,
    )
    by_key = {(r.construction, r.effect_size): r for r in rows}

    production = by_key[("production", 1.0)]
    uniform = by_key[("uniform_probe", 1.0)]

    # both grow in size
    assert production.joint_delta > 1.0
    assert uniform.joint_delta > 1.0

    # only the production construction rotates the joint trajectory
    assert production.joint_angle is not None and production.joint_angle > 1.0
    assert uniform.joint_angle is not None and uniform.joint_angle < 1e-3

    # every construction is size-pure *within* a block; that was never the issue
    assert production.max_block_angle < 1e-3
    assert uniform.max_block_angle < 1e-3
