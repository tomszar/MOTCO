from __future__ import annotations

import pandas as pd
import pytest

from motco.simulations import (
    InterSIMDependencyError,
    InterSIMParams,
    InterSIMResult,
    SemiSyntheticTrajectoryError,
    SemiSyntheticTrajectoryParams,
    generate_semisynthetic_trajectory,
    generate_semisynthetic_trajectory_from_intersim,
    semisynthetic,
)


def make_intersim_result(*, clusters: list[int | str] | None = None) -> InterSIMResult:
    if clusters is None:
        clusters = [2, 2, 2, 2, 1, 1, 1, 1, 3, 3, 3, 3]
    sample_ids = pd.Index([f"s{i}" for i in range(len(clusters))])
    methylation = pd.DataFrame(0.0, index=sample_ids, columns=["m0", "m1", "m2", "m3"])
    expression = pd.DataFrame(0.0, index=sample_ids, columns=["g0", "g1", "g2", "g3"])
    proteomics = pd.DataFrame(0.0, index=sample_ids, columns=["p0", "p1", "p2", "p3"])
    cluster_series = pd.Series(clusters, index=sample_ids, name="cluster")
    return InterSIMResult(
        methylation=methylation,
        expression=expression,
        proteomics=proteomics,
        sample_ids=sample_ids,
        clusters=cluster_series,
        metadata={"source": "fixture"},
    )


def explicit_params(
    mode: str = "none",
    *,
    effect_size: float = 1.0,
    seed: int = 7,
) -> SemiSyntheticTrajectoryParams:
    return SemiSyntheticTrajectoryParams(
        seed=seed,
        trajectory_mode=mode,  # type: ignore[arg-type]
        group_effect_size=effect_size,
        group_ratio=0.5,
        prop_affected_features=0.0,
        affected_features={
            "methylation": ["m0", "m1"],
            "expression": ["g0"],
            "proteomics": ["p0"],
        },
    )


def test_clusters_as_stages_and_original_clusters_preserved() -> None:
    dataset = generate_semisynthetic_trajectory(make_intersim_result(), explicit_params("none"))

    assert dataset.metadata["sample_id"].tolist() == [f"s{i}" for i in range(12)]
    assert dataset.metadata["cluster"].tolist() == [2, 2, 2, 2, 1, 1, 1, 1, 3, 3, 3, 3]
    assert dataset.metadata["stage"].tolist() == [1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2]
    assert dataset.truth["stage_mapping"] == {"1": 0, "2": 1, "3": 2}
    assert dataset.truth["stage_assumption"] == "clusters-as-stages"


def test_group_assignment_is_deterministic_within_stages() -> None:
    result = make_intersim_result()
    first = generate_semisynthetic_trajectory(result, explicit_params("none", seed=99))
    second = generate_semisynthetic_trajectory(result, explicit_params("none", seed=99))

    assert first.metadata["group"].tolist() == second.metadata["group"].tolist()
    for stage, frame in first.metadata.groupby("stage"):
        assert set(frame["group"]) == {"A", "B"}, stage
        assert frame["group"].value_counts().to_dict() == {"A": 2, "B": 2}


def test_insufficient_stage_size_is_rejected() -> None:
    result = make_intersim_result(clusters=[1, 2, 2, 3, 3])

    with pytest.raises(SemiSyntheticTrajectoryError, match="at least 2"):
        generate_semisynthetic_trajectory(result, explicit_params("none"))


def test_shape_mode_requires_at_least_three_stages() -> None:
    result = make_intersim_result(clusters=[1, 1, 1, 1, 2, 2, 2, 2])

    with pytest.raises(SemiSyntheticTrajectoryError, match="requires at least three stages"):
        generate_semisynthetic_trajectory(result, explicit_params("shape"))


def test_explicit_affected_features_are_honored() -> None:
    dataset = generate_semisynthetic_trajectory(make_intersim_result(), explicit_params("translation"))

    assert dataset.truth["affected_features"] == {
        "methylation": ["m0", "m1"],
        "expression": ["g0"],
        "proteomics": ["p0"],
    }


def test_proportion_based_affected_features_are_reproducible() -> None:
    params = SemiSyntheticTrajectoryParams(seed=12, prop_affected_features=0.5)
    first = generate_semisynthetic_trajectory(make_intersim_result(), params)
    second = generate_semisynthetic_trajectory(make_intersim_result(), params)

    assert first.truth["affected_features"] == second.truth["affected_features"]
    assert len(first.truth["affected_features"]["methylation"]) == 2
    assert len(first.truth["affected_features"]["expression"]) == 2
    assert len(first.truth["affected_features"]["proteomics"]) == 2


def test_unknown_explicit_affected_feature_is_rejected() -> None:
    params = SemiSyntheticTrajectoryParams(
        seed=1,
        affected_features={"methylation": ["not_a_feature"]},
    )

    with pytest.raises(SemiSyntheticTrajectoryError, match="unknown features"):
        generate_semisynthetic_trajectory(make_intersim_result(), params)


def test_invalid_affected_feature_proportion_is_rejected() -> None:
    params = SemiSyntheticTrajectoryParams(seed=1, prop_affected_features={"methylation": 1.5})

    with pytest.raises(SemiSyntheticTrajectoryError, match="between 0 and 1"):
        generate_semisynthetic_trajectory(make_intersim_result(), params)


def test_none_mode_preserves_omics_values() -> None:
    result = make_intersim_result()
    dataset = generate_semisynthetic_trajectory(result, explicit_params("none"))

    pd.testing.assert_frame_equal(dataset.methylation, result.methylation)
    pd.testing.assert_frame_equal(dataset.expression, result.expression)
    pd.testing.assert_frame_equal(dataset.proteomics, result.proteomics)


def test_zero_effect_preserves_omics_values() -> None:
    result = make_intersim_result()
    dataset = generate_semisynthetic_trajectory(result, explicit_params("magnitude", effect_size=0.0))

    pd.testing.assert_frame_equal(dataset.methylation, result.methylation)
    pd.testing.assert_frame_equal(dataset.expression, result.expression)
    pd.testing.assert_frame_equal(dataset.proteomics, result.proteomics)


@pytest.mark.parametrize(
    ("mode", "coefficients"),
    [
        ("translation", [1.0, 1.0, 1.0]),
        ("magnitude", [0.0, 1.0, 2.0]),
        ("orientation", [0.0, 1.0, 2.0]),
        ("shape", [0.0, 1.0, 0.0]),
    ],
)
def test_non_null_modes_shift_target_group_by_stage(mode: str, coefficients: list[float]) -> None:
    dataset = generate_semisynthetic_trajectory(make_intersim_result(), explicit_params(mode, effect_size=2.0))
    target = dataset.metadata["group"] == "B"

    assert (dataset.methylation.loc[~target, ["m0", "m1"]] == 0).all().all()
    assert (dataset.expression.loc[~target, ["g0"]] == 0).all().all()
    assert (dataset.proteomics.loc[~target, ["p0"]] == 0).all().all()
    assert (dataset.methylation[["m2", "m3"]] == 0).all().all()
    assert (dataset.expression[["g1", "g2", "g3"]] == 0).all().all()
    assert (dataset.proteomics[["p1", "p2", "p3"]] == 0).all().all()

    for stage, coefficient in enumerate(coefficients):
        stage_target = target & (dataset.metadata["stage"] == stage)
        expected = 2.0 * coefficient
        assert (dataset.expression.loc[stage_target, "g0"] == expected).all()
        assert (dataset.proteomics.loc[stage_target, "p0"] == expected).all()
        if mode in {"orientation", "shape"}:
            assert (dataset.methylation.loc[stage_target, "m0"] == expected).all()
            assert (dataset.methylation.loc[stage_target, "m1"] == -expected).all()
        else:
            assert (dataset.methylation.loc[stage_target, "m0"] == expected).all()
            assert (dataset.methylation.loc[stage_target, "m1"] == expected).all()

    assert dataset.truth["effect_coefficients"] == coefficients


def test_convenience_generation_calls_run_intersim(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_intersim(
        params: InterSIMParams,
        *,
        rscript: str = "Rscript",
        check_dependency: bool = True,
    ) -> InterSIMResult:
        assert params.seed == 123
        assert rscript == "custom-rscript"
        assert not check_dependency
        return make_intersim_result()

    monkeypatch.setattr(semisynthetic, "run_intersim", fake_run_intersim)

    dataset = generate_semisynthetic_trajectory_from_intersim(
        InterSIMParams(seed=123),
        explicit_params("none"),
        rscript="custom-rscript",
        check_dependency=False,
    )

    assert dataset.metadata.shape[0] == 12


def test_convenience_generation_propagates_intersim_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_intersim(
        params: InterSIMParams,
        *,
        rscript: str = "Rscript",
        check_dependency: bool = True,
    ) -> InterSIMResult:
        raise InterSIMDependencyError("missing InterSIM")

    monkeypatch.setattr(semisynthetic, "run_intersim", fake_run_intersim)

    with pytest.raises(InterSIMDependencyError, match="missing InterSIM"):
        generate_semisynthetic_trajectory_from_intersim(
            InterSIMParams(seed=123),
            explicit_params("none"),
        )
