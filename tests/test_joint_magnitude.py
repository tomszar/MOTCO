"""The ``joint`` magnitude construction and the invariants its adoption carries.

``magnitude_kind='all'`` scales methylation's delta alone, so the concatenated
trajectory rotates even though each omic block on its own is exactly
size-scaled. ``'joint'`` scales every omic's delta together. It is a production
value — selectable from the generator parameters, a study configuration, and
the CLI — but a *sibling* of ``'all'``: the default is unchanged, and every
dataset and parameter signature that does not select ``joint`` is byte-identical
to what it was before the value existed.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from motco.simulations.semisynthetic import (
    SemiSyntheticTrajectoryError,
    SemiSyntheticTrajectoryParams,
    generate_semisynthetic_trajectory,
    load_reference,
)
from motco.simulations.study.config import StudyConfigError, load_study_config


@pytest.fixture(scope="module")
def reference():
    return load_reference()


def _params(mode: str = "magnitude", **overrides) -> SemiSyntheticTrajectoryParams:
    base = dict(seed=7, trajectory_mode=mode, n_samples=240, n_stages=3, group_effect_size=0.6)
    base.update(overrides)
    return SemiSyntheticTrajectoryParams(**base)  # type: ignore[arg-type]


def _assert_datasets_equal(left, right) -> None:
    for name in ("methylation", "expression", "proteomics", "metadata"):
        a, b = getattr(left, name), getattr(right, name)
        assert list(a.index) == list(b.index) and list(a.columns) == list(b.columns)
        np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())


# --------------------------------------------------------------------------- #
# joint is selectable, and records what it did
# --------------------------------------------------------------------------- #


def test_joint_scales_every_delta_and_keeps_indicators(reference) -> None:
    dataset = generate_semisynthetic_trajectory(
        _params(group_effect_size=1.0, magnitude_kind="joint"), reference=reference
    )
    truth = dataset.truth
    assert truth["magnitude_kind"] == "joint"
    assert truth["transform"] == {
        "magnitude_kind": "joint",
        "delta_methyl_scale": 2.0,
        "delta_expr_scale": 2.0,
        "delta_protein_scale": 2.0,
    }
    a, b = truth["deltas"]["A"], truth["deltas"]["B"]
    assert b == [2 * x for x in a]
    ind = truth["indicators"]
    for layer in ("methylation", "expression", "proteomics"):
        np.testing.assert_array_equal(ind["A"][layer], ind["B"][layer])


def test_joint_scale_tracks_the_effect_size(reference) -> None:
    dataset = generate_semisynthetic_trajectory(
        _params(group_effect_size=0.6, magnitude_kind="joint", delta_expr=1.5, delta_protein=3.0),
        reference=reference,
    )
    a, b = dataset.truth["deltas"]["A"], dataset.truth["deltas"]["B"]
    np.testing.assert_allclose(b, np.asarray(a) * 1.6)


def test_study_config_accepts_joint_and_refuses_unknown_kinds(tmp_path) -> None:
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

    for kind in ("all", "extremes", "joint"):
        path = tmp_path / f"{kind}.json"
        path.write_text(json.dumps(config(kind)), encoding="utf-8")
        assert load_study_config(path).generator.magnitude_kind == kind

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(config("methylation_only")), encoding="utf-8")
    with pytest.raises(StudyConfigError, match="generator.magnitude_kind"):
        load_study_config(bad)


def test_unknown_magnitude_kind_is_refused_at_generation(reference) -> None:
    params = _params(magnitude_kind="methylation_only")
    with pytest.raises(SemiSyntheticTrajectoryError, match="Unknown magnitude_kind"):
        generate_semisynthetic_trajectory(params, reference=reference)


@pytest.mark.parametrize("mode", ["none", "translation", "orientation", "shape"])
def test_joint_outside_magnitude_mode_is_inert(reference, mode) -> None:
    """No non-magnitude mode reads ``magnitude_kind``, so ``joint`` changes nothing."""

    kwargs = dict(group_effect_size=0.2) if mode != "none" else {}
    with_joint = generate_semisynthetic_trajectory(
        _params(mode, magnitude_kind="joint", **kwargs), reference=reference
    )
    with_all = generate_semisynthetic_trajectory(
        _params(mode, magnitude_kind="all", **kwargs), reference=reference
    )
    _assert_datasets_equal(with_joint, with_all)
    assert with_joint.truth["deltas"] == with_all.truth["deltas"]
    assert with_joint.truth["transform"] == with_all.truth["transform"]


# --------------------------------------------------------------------------- #
# sibling value: nothing that does not select joint moves
# --------------------------------------------------------------------------- #


def test_joint_at_zero_effect_equals_none(reference) -> None:
    joint = generate_semisynthetic_trajectory(
        _params(group_effect_size=0.0, magnitude_kind="joint"), reference=reference
    )
    none = generate_semisynthetic_trajectory(
        _params("none", group_effect_size=0.0), reference=reference
    )
    _assert_datasets_equal(joint, none)
    assert joint.truth["transform"] == {} == none.truth["transform"]
    assert joint.truth["deltas"] == none.truth["deltas"]
    assert joint.truth["indicator_counts"] == none.truth["indicator_counts"]


# Block sums and three sampled entries per block, computed at revision f90230c
# (before ``joint`` existed) with seed=7, n_samples=240, n_stages=3,
# group_effect_size=0.6. Pinned at a relative tolerance of 1e-9: a byte-level
# digest would be machine-specific (BLAS/CPU last-bit differences), whereas any
# change to the RNG call sequence moves these values at order one.
_PRE_JOINT_BLOCK_STATS = {
    "all": {
        "methylation": (24750.791527663598, 0.04542578069960075, 0.04370102732882042, 0.02840071546721136),
        "expression": (26266.179618364564, -0.7122378696242087, 0.6074954636986026, 0.6264035808155781),
        "proteomics": (31468.24446498954, -0.7131403871940891, 0.048633068616348446, 0.7131239345683809),
    },
    "extremes": {
        "methylation": (24217.696802519327, 0.04542578069960075, 0.04370102732882042, 0.02840071546721136),
        "expression": (26266.179618364564, -0.7122378696242087, 0.6074954636986026, 0.6264035808155781),
        "proteomics": (31468.24446498954, -0.7131403871940891, 0.048633068616348446, 0.7131239345683809),
    },
}


def _block_stats(dataset) -> dict[str, tuple[float, float, float, float]]:
    out = {}
    for name in ("methylation", "expression", "proteomics"):
        a = getattr(dataset, name).to_numpy()
        out[name] = (float(a.sum()), float(a[0, 0]), float(a[-1, -1]), float(a[117, 5]))
    return out


@pytest.mark.parametrize("kind", sorted(_PRE_JOINT_BLOCK_STATS))
def test_existing_magnitude_kinds_are_unchanged(reference, kind) -> None:
    dataset = generate_semisynthetic_trajectory(_params(magnitude_kind=kind), reference=reference)
    observed = _block_stats(dataset)
    for name, expected in _PRE_JOINT_BLOCK_STATS[kind].items():
        np.testing.assert_allclose(observed[name], expected, rtol=1e-9, atol=0.0, err_msg=f"{kind}/{name}")


# Parameter signatures of the Phase 5 paper-grade profile's anchor and
# magnitude cells, computed at revision f90230c. A change here would mean a
# committed record set could no longer be resumed or compared.
_PHASE5_SIGNATURES = {
    ("none", 0.0): "40bfad59c97678be6feb28f5317186ce183df1097544750d347f3286dabdfe28",
    ("magnitude", 0.25): "efe40757bac034a0d0151bbc26a2c5467449b8410ab0980fb634a0899e87a538",
    ("magnitude", 0.5): "e7da4a67d178d35f7ef24eafaba070a564b3ed71f3a6f969004ce905fd20113a",
    ("magnitude", 0.75): "186e78c7ddbc0d4b77ce502888477a6401c10c0d9ad97fca6b604dd5c2ed91dc",
    ("magnitude", 1.0): "cd7e28a176fdfb791bdb1e6919d96fd79382fb9c6ea7cbaa5d7b5606857abdf5",
}


def test_phase5_parameter_signatures_are_unchanged() -> None:
    from motco.simulations.grid import parameter_signature
    from motco.simulations.study.enumerate import enumerate_study

    config = load_study_config("examples/trajectory_power_study/phase5_power_study.json")
    seen = {}
    for cell in enumerate_study(config).cells:
        g = cell.generator_params
        if cell.phase == "power_primary" and (g.trajectory_mode, g.group_effect_size) in _PHASE5_SIGNATURES:
            seen[(g.trajectory_mode, g.group_effect_size)] = parameter_signature(cell)
    assert seen == _PHASE5_SIGNATURES


# --------------------------------------------------------------------------- #
# the finding that motivated the value
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_joint_is_size_pure_in_the_joint_space():
    """Scaling every delta keeps the concatenated trajectory size-only.

    ``all`` rotates the concatenated trajectory (joint angle grows with the
    effect); ``joint`` leaves it at the floating-point floor.
    """

    from motco.simulations.specificity import compare_uniform_delta_construction

    rows = compare_uniform_delta_construction(
        effect_sizes=(0.0, 1.0),
        n_samples=180,
        n_stages=3,
        p_dmp=0.1,
    )
    by_key = {(r.construction, r.effect_size): r for r in rows}

    production = by_key[("all", 1.0)]
    joint = by_key[("joint", 1.0)]

    # both grow in size
    assert production.joint_delta > 1.0
    assert joint.joint_delta > 1.0

    # only the production construction rotates the joint trajectory
    assert production.joint_angle is not None and production.joint_angle > 1.0
    assert joint.joint_angle is not None and joint.joint_angle < 1e-3

    # every construction is size-pure *within* a block; that was never the issue
    assert production.max_block_angle < 1e-3
    assert joint.max_block_angle < 1e-3
