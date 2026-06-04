from __future__ import annotations

import numpy as np
import pytest

from motco.simulations import (
    SemiSyntheticTrajectoryError,
    SemiSyntheticTrajectoryParams,
    generate_semisynthetic_trajectory,
    load_reference,
)


@pytest.fixture(scope="module")
def reference():
    return load_reference()


def make_params(mode: str = "none", **overrides) -> SemiSyntheticTrajectoryParams:
    base = dict(
        seed=7,
        trajectory_mode=mode,
        n_samples=240,
        n_stages=3,
        group_effect_size=0.6,
        group_ratio=0.5,
    )
    base.update(overrides)
    return SemiSyntheticTrajectoryParams(**base)  # type: ignore[arg-type]


def test_dataset_structure_and_feature_counts(reference) -> None:
    dataset = generate_semisynthetic_trajectory(make_params("orientation"), reference=reference)

    assert dataset.methylation.shape == (240, 367)
    assert dataset.expression.shape == (240, 131)
    assert dataset.proteomics.shape == (240, 160)
    assert list(dataset.metadata.columns) == ["sample_id", "group", "stage", "cluster"]
    assert sorted(dataset.metadata["group"].unique()) == ["A", "B"]
    assert sorted(dataset.metadata["stage"].unique()) == [0, 1, 2]
    # methylation is bounded in (0, 1)
    assert dataset.methylation.to_numpy().min() > 0
    assert dataset.methylation.to_numpy().max() < 1


def test_generation_is_reproducible(reference) -> None:
    first = generate_semisynthetic_trajectory(make_params("shape"), reference=reference)
    second = generate_semisynthetic_trajectory(make_params("shape"), reference=reference)

    np.testing.assert_array_equal(first.methylation.to_numpy(), second.methylation.to_numpy())
    np.testing.assert_array_equal(first.expression.to_numpy(), second.expression.to_numpy())
    assert first.metadata["group"].tolist() == second.metadata["group"].tolist()


def test_group_assignment_is_balanced_within_stages(reference) -> None:
    dataset = generate_semisynthetic_trajectory(make_params("none"), reference=reference)
    for _, frame in dataset.metadata.groupby("stage"):
        assert set(frame["group"]) == {"A", "B"}


def test_truth_records_mode_deltas_and_indicators(reference) -> None:
    dataset = generate_semisynthetic_trajectory(make_params("magnitude"), reference=reference)
    truth = dataset.truth

    assert truth["trajectory_mode"] == "magnitude"
    assert truth["group_effect_size"] == 0.6
    assert truth["n_stages"] == 3
    assert set(truth["indicator_counts"]) == {"A", "B"}
    assert set(truth["indicators"]["A"]) == {"methylation", "expression", "proteomics"}
    # indicator arrays are (n_feat, n_stages)
    assert truth["indicators"]["A"]["methylation"].shape == (367, 3)


def test_none_mode_uses_identical_group_indicators(reference) -> None:
    dataset = generate_semisynthetic_trajectory(make_params("none"), reference=reference)
    ind = dataset.truth["indicators"]
    np.testing.assert_array_equal(ind["A"]["methylation"], ind["B"]["methylation"])
    np.testing.assert_array_equal(ind["A"]["expression"], ind["B"]["expression"])
    assert dataset.truth["deltas"]["A"] == dataset.truth["deltas"]["B"]


def test_magnitude_scales_only_methylation_delta_and_keeps_indicators(reference) -> None:
    dataset = generate_semisynthetic_trajectory(
        make_params("magnitude", group_effect_size=1.0), reference=reference
    )
    ind = dataset.truth["indicators"]
    np.testing.assert_array_equal(ind["A"]["methylation"], ind["B"]["methylation"])
    deltas = dataset.truth["deltas"]
    # only methylation delta scales: delta_methyl_B = (1 + e) * delta_methyl, e = 1.0
    assert deltas["B"][0] == 2 * deltas["A"][0]
    assert deltas["B"][1:] == deltas["A"][1:]


def test_magnitude_kind_defaults_to_all_and_records_truth(reference) -> None:
    dataset = generate_semisynthetic_trajectory(
        make_params("magnitude", group_effect_size=1.0), reference=reference
    )
    # Default is the all-stages variant; truth records it both top-level and in transform.
    assert dataset.truth["magnitude_kind"] == "all"
    assert dataset.truth["transform"]["magnitude_kind"] == "all"
    assert dataset.truth["transform"]["delta_methyl_scale"] == 2.0


def test_magnitude_extremes_scales_only_endpoint_indicators(reference) -> None:
    dataset = generate_semisynthetic_trajectory(
        make_params("magnitude", n_stages=4, group_effect_size=1.0, magnitude_kind="extremes"),
        reference=reference,
    )
    truth = dataset.truth
    assert truth["magnitude_kind"] == "extremes"
    assert truth["transform"] == {"magnitude_kind": "extremes", "magnitude_scale": 2.0}
    # deltas are untouched (the scale lives in the indicators, not the global delta)
    assert truth["deltas"]["A"] == truth["deltas"]["B"]
    a_m, b_m = truth["indicators"]["A"]["methylation"], truth["indicators"]["B"]["methylation"]
    # endpoints scaled by (1 + e) = 2; interior stages untouched
    np.testing.assert_allclose(b_m[:, 0], a_m[:, 0] * 2.0)
    np.testing.assert_allclose(b_m[:, -1], a_m[:, -1] * 2.0)
    np.testing.assert_array_equal(b_m[:, 1:-1], a_m[:, 1:-1])


def test_orientation_preserves_methylation_cardinality_per_stage(reference) -> None:
    dataset = generate_semisynthetic_trajectory(
        make_params("orientation", group_effect_size=1.0), reference=reference
    )
    counts = dataset.truth["indicator_counts"]
    # relocation preserves per-stage methylation cardinality (same number of sites)
    assert counts["A"]["methylation"] == counts["B"]["methylation"]
    # but the sites moved, so the actual indicators differ
    ind = dataset.truth["indicators"]
    assert not np.array_equal(ind["A"]["methylation"], ind["B"]["methylation"])
    assert dataset.truth["transform"]["orientation_relocated"] > 0


def test_shape_fixes_endpoints_and_perturbs_interior(reference) -> None:
    dataset = generate_semisynthetic_trajectory(
        make_params("shape", n_stages=4, group_effect_size=1.0), reference=reference
    )
    ind = dataset.truth["indicators"]
    a_m, b_m = ind["A"]["methylation"], ind["B"]["methylation"]
    # endpoints (first and last stage) are unchanged
    np.testing.assert_array_equal(a_m[:, 0], b_m[:, 0])
    np.testing.assert_array_equal(a_m[:, -1], b_m[:, -1])
    # at least one interior stage is changed
    assert not np.array_equal(a_m[:, 1:-1], b_m[:, 1:-1])


def test_translation_adds_disjoint_constant_set(reference) -> None:
    dataset = generate_semisynthetic_trajectory(make_params("translation"), reference=reference)
    ind = dataset.truth["indicators"]
    a_m, b_m = ind["A"]["methylation"], ind["B"]["methylation"]
    # group B keeps A's stage sites and adds an extra constant set U at every stage
    n_extra = dataset.truth["transform"]["translation_set_size"]
    assert n_extra > 0
    # the extra set is present in every B stage and absent from every A stage
    extra_rows = (b_m.sum(1) > 0) & (a_m.sum(1) == 0)
    assert int(extra_rows.sum()) == n_extra
    assert np.all(b_m[extra_rows].sum(1) == dataset.truth["n_stages"])  # on at every stage
    # A's stage-changing sites are unchanged in B
    stage_rows = a_m.sum(1) > 0
    np.testing.assert_array_equal(a_m[stage_rows], b_m[stage_rows])


def test_none_mode_has_empty_transform(reference) -> None:
    dataset = generate_semisynthetic_trajectory(make_params("none"), reference=reference)
    assert dataset.truth["transform"] == {}


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"trajectory_mode": "bogus"}, "Unknown trajectory_mode"),
        ({"magnitude_kind": "bogus"}, "Unknown magnitude_kind"),
        ({"group_labels": ("A", "A")}, "two distinct labels"),
        ({"group_ratio": 1.5}, "group_ratio"),
        ({"group_effect_size": -1.0}, "non-negative"),
        ({"p_dmp": 2.0}, "p_dmp"),
        ({"n_stages": 1}, "n_stages"),
        ({"delta_expr": -1.0}, "delta_expr"),
    ],
)
def test_invalid_params_are_rejected(reference, overrides, match) -> None:
    with pytest.raises(SemiSyntheticTrajectoryError, match=match):
        generate_semisynthetic_trajectory(make_params("none", **overrides), reference=reference)


def test_shape_mode_requires_three_stages(reference) -> None:
    with pytest.raises(SemiSyntheticTrajectoryError, match="at least three stages"):
        generate_semisynthetic_trajectory(make_params("shape", n_stages=2), reference=reference)


def test_stage_sample_prop_must_match_n_stages(reference) -> None:
    with pytest.raises(SemiSyntheticTrajectoryError, match="one entry per stage"):
        generate_semisynthetic_trajectory(
            make_params("none", stage_sample_prop=(0.5, 0.5)), reference=reference
        )
