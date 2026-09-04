from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from motco.stats import (
    AttributionError,
    analyze_orientation_attribution,
    attribution_frames,
    configuration_spectrum,
    fit_plsda_model,
    principal_orientation,
    write_attribution_outputs,
)
from motco.stats.trajectory import _estimate_orientation


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[np.ndarray] = []
    metadata: list[tuple[str, str]] = []
    base = {
        ("A", "t0"): [0.0, 0.0, 0.0],
        ("A", "t1"): [1.0, 0.0, 1.0],
        ("A", "t2"): [2.0, 1.0, 1.0],
        ("B", "t0"): [0.0, 1.0, 0.0],
        ("B", "t1"): [1.0, 2.0, 1.0],
        ("B", "t2"): [2.0, 3.0, 2.0],
    }
    for group in ("A", "B"):
        for stage in ("t0", "t1", "t2"):
            for replicate in range(3):
                rows.append(np.asarray(base[(group, stage)]) + replicate * 0.01)
                metadata.append((group, stage))
    index = pd.Index([f"sample-{i}" for i in range(len(rows))])
    X = pd.DataFrame(rows, index=index, columns=["f1", "f2", "f3"])
    meta = pd.DataFrame(metadata, index=index, columns=["group", "stage"])
    return X, meta


def _model(X: pd.DataFrame, meta: pd.DataFrame):
    return fit_plsda_model(X, meta["group"], n_components=2)


def test_multistage_attribution_preserves_order_and_raw_vectors() -> None:
    X, metadata = _inputs()
    result = analyze_orientation_attribution(X, metadata, _model(X, metadata), bootstrap_replicates=0)

    assert result.groups == ("A", "B")
    assert result.stages == ("t0", "t1", "t2")
    assert [record.transition_id for record in result.transitions] == ["t0->t1", "t1->t2"]
    assert len(result.transition_vectors) == 2 * 3 * 2 * 3
    assert set(result.feature_effects["effect_unit"]) == {"standardized_input"}
    assert "p_value" not in result.feature_effects.columns
    first = result.means.query("group == 'A' and stage == 't1'").iloc[0]
    np.testing.assert_allclose(first[["f1", "f2", "f3"]].to_numpy(dtype=float), [1.01, 0.01, 1.01])


def test_explicit_stage_order_and_precomputed_means() -> None:
    X, metadata = _inputs()
    model = _model(X, metadata)
    means = (
        X.assign(group=metadata["group"], stage=metadata["stage"])
        .groupby(["group", "stage"])[["f1", "f2", "f3"]]
        .mean()
    )
    result = analyze_orientation_attribution(
        X,
        metadata,
        model,
        groups=["B", "A"],
        stages=["t2", "t1", "t0"],
        precomputed_means=means,
    )
    assert result.config.mean_source == "precomputed"
    assert result.groups == ("B", "A")
    assert result.stages == ("t2", "t1", "t0")


def test_zero_transition_is_explicitly_unavailable() -> None:
    X, metadata = _inputs()
    mask = (metadata["group"] == "A") & (metadata["stage"] == "t0")
    X.loc[mask, :] = [4.0, 4.0, 4.0]
    mask = (metadata["group"] == "A") & (metadata["stage"] == "t1")
    X.loc[mask, :] = [4.0, 4.0, 4.0]
    result = analyze_orientation_attribution(X, metadata, _model(X, metadata))

    observed = result.transitions[0].observed
    assert observed.path_lengths[0] == pytest.approx(0.0)
    assert observed.unit_directions[0] is None
    assert observed.directional_contrast is None
    assert result.feature_effects.query("transition_id == 't0->t1' and component == 'observed'")[
        "effect_standardized"
    ].isna().all()


def test_reconstruction_residual_and_original_units_are_separate() -> None:
    X, metadata = _inputs()
    result = analyze_orientation_attribution(
        X,
        metadata,
        _model(X, metadata),
        original_scales=pd.Series([2.0, 3.0, 4.0], index=["f1", "f2", "f3"]),
    )
    observed = result.transitions[0].observed
    captured = result.transitions[0].pls_captured
    residual = result.transitions[0].residual
    np.testing.assert_allclose(
        residual.group_transitions[0], observed.group_transitions[0] - captured.group_transitions[0]
    )
    np.testing.assert_allclose(
        residual.group_transitions[1], observed.group_transitions[1] - captured.group_transitions[1]
    )
    np.testing.assert_allclose(
        result.feature_effects.query("component == 'observed'")["effect_original"].to_numpy()[:3],
        result.feature_effects.query("component == 'observed'")["effect_standardized"].to_numpy()[:3]
        * np.array([2.0, 3.0, 4.0]),
    )
    assert result.config.model_components == 2


def test_bootstrap_is_deterministic_and_reports_stability() -> None:
    X, metadata = _inputs()
    model = _model(X, metadata)
    first = analyze_orientation_attribution(
        X, metadata, model, bootstrap_replicates=12, bootstrap_seed=19, top_k=2
    )
    second = analyze_orientation_attribution(
        X, metadata, model, bootstrap_replicates=12, bootstrap_seed=19, top_k=2
    )
    pd.testing.assert_frame_equal(first.bootstrap_summaries, second.bootstrap_summaries)
    assert first.bootstrap_summaries["valid_replicates"].max() == 12
    assert first.bootstrap_summaries["sign_stability"].notna().any()
    disabled = analyze_orientation_attribution(X, metadata, model)
    assert disabled.bootstrap_summaries["valid_replicates"].eq(0).all()
    assert disabled.bootstrap_summaries["sign_stability"].isna().all()


def test_validation_for_alignment_dimensions_scales_and_labels() -> None:
    X, metadata = _inputs()
    model = _model(X, metadata)
    with pytest.raises(AttributionError, match="aligned"):
        analyze_orientation_attribution(X, metadata.sample(frac=1.0, random_state=1), model)
    with pytest.raises(AttributionError, match="expects 3 features"):
        analyze_orientation_attribution(X.iloc[:, :2], metadata, model)
    with pytest.raises(AttributionError, match="feature order"):
        analyze_orientation_attribution(
            X, metadata, model, original_scales=pd.Series([1.0, 2.0, 3.0], index=["f2", "f1", "f3"])
        )
    with pytest.raises(AttributionError, match="omits"):
        analyze_orientation_attribution(X, metadata, model, feature_groups={"f1": "m1", "f2": "m1"})
    with pytest.raises(AttributionError, match="unknown feature"):
        analyze_orientation_attribution(
            X,
            metadata,
            model,
            feature_groups={"f1": "m1", "f2": "m1", "f3": "m2", "unknown": "m3"},
        )


def test_module_aggregation_and_report_views(tmp_path) -> None:
    X, metadata = _inputs()
    result = analyze_orientation_attribution(
        X,
        metadata,
        _model(X, metadata),
        feature_groups={"f1": "module-a", "f2": "module-a", "f3": "module-b"},
        label_namespace="pathway",
    )
    assert set(result.aggregate_effects["label"]) == {"module-a", "module-b"}
    assert set(result.aggregate_effects["label_source"]) == {"caller-supplied"}
    # two transitions plus the principal-orientation block, three components, two labels
    assert len(result.aggregate_effects) == 3 * 3 * 2
    frames = attribution_frames(result)
    assert {
        "feature_effects",
        "transition_summaries",
        "aggregate_effects",
        "principal_orientation",
        "principal_orientation_vectors",
        "configuration",
        "interpretation",
    } <= set(frames)
    paths = write_attribution_outputs(result, tmp_path / "attribution")
    assert (tmp_path / "attribution" / "feature_effects.csv").exists()
    assert paths["interpretation"].read_text(encoding="utf-8").find("causal_boundary") >= 0


# --- shared orientation functional --------------------------------------------


def test_public_orientation_functional_matches_the_tested_estimator() -> None:
    rng = np.random.default_rng(11)
    configuration = rng.normal(size=(4, 6))
    np.testing.assert_array_equal(
        principal_orientation(configuration),
        _estimate_orientation(configuration, [0, 1, 2, 3]),
    )
    subset = [3, 1, 0]
    np.testing.assert_array_equal(
        principal_orientation(configuration, levels=subset),
        _estimate_orientation(configuration, subset),
    )


def test_public_orientation_functional_rejects_degenerate_shapes() -> None:
    with pytest.raises(ValueError, match="2-D matrix"):
        principal_orientation(np.zeros(3))
    with pytest.raises(ValueError, match="at least two stages"):
        principal_orientation(np.zeros((1, 3)))


def test_configuration_eigengap_separates_straight_bent_and_isotropic() -> None:
    straight = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    assert configuration_spectrum(straight)["relative_eigengap"] == pytest.approx(1.0)

    bent = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.6], [3.0, 1.8]])
    bent_gap = configuration_spectrum(bent)["relative_eigengap"]
    assert 0.05 < bent_gap < 1.0

    isotropic = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    assert configuration_spectrum(isotropic)["relative_eigengap"] == pytest.approx(0.0)


# --- principal-orientation component ------------------------------------------


def _two_stage_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[np.ndarray] = []
    metadata: list[tuple[str, str]] = []
    base = {
        ("A", "t0"): [0.0, 0.0, 0.0],
        ("A", "t1"): [1.0, 0.0, 1.0],
        ("B", "t0"): [0.0, 1.0, 0.0],
        ("B", "t1"): [1.0, 2.0, 1.0],
    }
    for group in ("A", "B"):
        for stage in ("t0", "t1"):
            for replicate in range(3):
                rows.append(np.asarray(base[(group, stage)]) + replicate * 0.01)
                metadata.append((group, stage))
    index = pd.Index([f"sample-{i}" for i in range(len(rows))])
    X = pd.DataFrame(rows, index=index, columns=["f1", "f2", "f3"])
    meta = pd.DataFrame(metadata, index=index, columns=["group", "stage"])
    return X, meta


def _bent_inputs(seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two groups whose four-stage trajectories bend in different planes."""

    rng = np.random.default_rng(seed)
    base = {
        ("A", "t0"): [0.0, 0.0, 0.0, 0.0],
        ("A", "t1"): [1.0, 0.8, 0.0, 0.0],
        ("A", "t2"): [2.0, 0.4, 0.3, 0.0],
        ("A", "t3"): [3.0, 0.0, 0.0, 0.2],
        ("B", "t0"): [0.0, 0.0, 0.0, 0.0],
        ("B", "t1"): [0.6, 0.0, 1.1, 0.0],
        ("B", "t2"): [1.4, 0.2, 1.6, 0.4],
        ("B", "t3"): [2.2, 0.1, 2.4, 0.9],
    }
    rows: list[np.ndarray] = []
    metadata: list[tuple[str, str]] = []
    for group in ("A", "B"):
        for stage in ("t0", "t1", "t2", "t3"):
            for _ in range(5):
                rows.append(np.asarray(base[(group, stage)]) + rng.normal(scale=0.01, size=4))
                metadata.append((group, stage))
    index = pd.Index([f"sample-{i}" for i in range(len(rows))])
    X = pd.DataFrame(rows, index=index, columns=["f1", "f2", "f3", "f4"])
    meta = pd.DataFrame(metadata, index=index, columns=["group", "stage"])
    return X, meta


def test_principal_orientation_block_is_populated_and_configured() -> None:
    X, metadata = _inputs()
    result = analyze_orientation_attribution(X, metadata, _model(X, metadata))
    principal = result.principal_orientation

    assert principal.component_id == "principal"
    assert (principal.from_stage, principal.to_stage) == ("t0", "t2")
    assert result.config.principal_component_id == "principal"
    assert result.config.eigengap_threshold == pytest.approx(0.05)
    assert principal.orientation_available == (True, True)
    for component_name in ("observed", "pls_captured", "residual"):
        component = getattr(principal, component_name)
        assert component.directional_contrast is not None
        assert component.directional_contrast.shape == (3,)
    for axis in principal.observed.unit_directions:
        assert axis is not None
        assert float(np.linalg.norm(axis)) == pytest.approx(1.0)
    assert [record.transition_id for record in result.transitions] == ["t0->t1", "t1->t2"]
    assert "principal" in set(result.feature_effects["transition_id"])


def test_principal_axes_match_the_tested_estimator_per_group() -> None:
    X, metadata = _bent_inputs()
    result = analyze_orientation_attribution(X, metadata, _model(X, metadata))
    means = result.means
    for group_index, group in enumerate(result.groups):
        configuration = (
            means.query("group == @group")
            .set_index("stage")
            .loc[list(result.stages), list(result.feature_names)]
            .to_numpy(dtype=float)
        )
        axis = result.principal_orientation.observed.unit_directions[group_index]
        assert axis is not None
        np.testing.assert_allclose(
            axis, _estimate_orientation(configuration, list(range(len(result.stages))))
        )


def test_multistage_principal_contrast_differs_from_every_transition() -> None:
    X, metadata = _bent_inputs()
    result = analyze_orientation_attribution(X, metadata, _model(X, metadata))
    principal_contrast = result.principal_orientation.observed.directional_contrast
    assert principal_contrast is not None
    for transition in result.transitions:
        contrast = transition.observed.directional_contrast
        assert contrast is not None
        assert not np.allclose(principal_contrast, contrast, atol=1e-6)


def test_two_stage_principal_contrast_equals_the_single_transition() -> None:
    X, metadata = _two_stage_inputs()
    result = analyze_orientation_attribution(X, metadata, _model(X, metadata))
    assert len(result.transitions) == 1
    transition = result.transitions[0]
    principal = result.principal_orientation
    for component_name in ("observed", "pls_captured", "residual"):
        expected = getattr(transition, component_name).directional_contrast
        actual = getattr(principal, component_name).directional_contrast
        assert expected is not None and actual is not None
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


def test_near_isotropic_configuration_is_flagged_degenerate() -> None:
    X, metadata = _inputs()
    square = {
        ("A", "t0"): [1.0, 0.0, 0.0],
        ("A", "t1"): [0.0, 1.0, 0.0],
        ("A", "t2"): [-1.0, 0.0, 0.0],
        ("A", "t3"): [0.0, -1.0, 0.0],
    }
    rows: list[np.ndarray] = []
    meta_rows: list[tuple[str, str]] = []
    for group in ("A", "B"):
        for stage_index, stage in enumerate(("t0", "t1", "t2", "t3")):
            for replicate in range(3):
                offset = 0.0 if group == "A" else 0.5
                point = np.asarray(square[("A", stage)]) + offset * np.asarray([0.0, 0.0, stage_index])
                rows.append(point + replicate * 0.001)
                meta_rows.append((group, stage))
    index = pd.Index([f"sample-{i}" for i in range(len(rows))])
    X = pd.DataFrame(rows, index=index, columns=["f1", "f2", "f3"])
    metadata = pd.DataFrame(meta_rows, index=index, columns=["group", "stage"])
    result = analyze_orientation_attribution(X, metadata, _model(X, metadata))
    principal = result.principal_orientation

    assert principal.degenerate
    assert any(source.startswith("observed:A") for source in principal.degeneracy_sources)
    # A degenerate contrast is still reported, flagged rather than dropped.
    assert principal.observed.contrast_available
    frame = result.frames()["principal_orientation"]
    assert frame["degenerate"].all()
    assert frame["relative_eigengap_observed_group_1"].iloc[0] < 0.05

    permissive = analyze_orientation_attribution(
        X, metadata, _model(X, metadata), eigengap_threshold=0.0
    )
    assert not permissive.principal_orientation.degenerate
    assert permissive.config.eigengap_threshold == pytest.approx(0.0)


def test_closed_trajectory_marks_the_principal_orientation_unavailable() -> None:
    X, metadata = _inputs()
    mask = (metadata["group"] == "A") & (metadata["stage"] == "t2")
    X.loc[mask, :] = X.loc[(metadata["group"] == "A") & (metadata["stage"] == "t0"), :].to_numpy()
    result = analyze_orientation_attribution(X, metadata, _model(X, metadata))
    principal = result.principal_orientation

    assert principal.orientation_available == (False, True)
    assert principal.observed.unit_directions[0] is None
    assert principal.observed.directional_contrast is None
    assert not principal.observed.contrast_available
    assert (
        result.feature_effects.query("transition_id == 'principal' and component == 'observed'")[
            "effect_standardized"
        ]
        .isna()
        .all()
    )


def test_principal_component_id_must_not_collide_with_a_transition() -> None:
    X, metadata = _inputs()
    model = _model(X, metadata)
    with pytest.raises(AttributionError, match="collides with stage-derived transition id"):
        analyze_orientation_attribution(X, metadata, model, principal_component_id="t0->t1")
    with pytest.raises(AttributionError, match="non-empty string"):
        analyze_orientation_attribution(X, metadata, model, principal_component_id="  ")
    with pytest.raises(AttributionError, match=r"\[0, 1\]"):
        analyze_orientation_attribution(X, metadata, model, eigengap_threshold=1.5)

    renamed = analyze_orientation_attribution(X, metadata, model, principal_component_id="axis")
    assert renamed.config.principal_component_id == "axis"
    assert "axis" in set(renamed.feature_effects["transition_id"])
    assert set(renamed.bootstrap_summaries["transition_id"]) == {"t0->t1", "t1->t2", "axis"}


def test_principal_effects_convert_to_original_units() -> None:
    X, metadata = _inputs()
    scales = np.array([2.0, 3.0, 4.0])
    result = analyze_orientation_attribution(
        X,
        metadata,
        _model(X, metadata),
        original_scales=pd.Series(scales, index=["f1", "f2", "f3"]),
    )
    block = result.feature_effects.query("transition_id == 'principal'")
    assert len(block) == 3 * 3
    np.testing.assert_allclose(
        block["effect_original"].to_numpy(dtype=float),
        block["effect_standardized"].to_numpy(dtype=float) * np.tile(scales, 3),
    )


def test_principal_bootstrap_is_reproducible_and_sign_anchored() -> None:
    X, metadata = _bent_inputs()
    model = _model(X, metadata)
    first = analyze_orientation_attribution(
        X, metadata, model, bootstrap_replicates=12, bootstrap_seed=19, top_k=2
    )
    second = analyze_orientation_attribution(
        X, metadata, model, bootstrap_replicates=12, bootstrap_seed=19, top_k=2
    )
    pd.testing.assert_frame_equal(first.bootstrap_summaries, second.bootstrap_summaries)
    principal_rows = first.bootstrap_summaries.query("transition_id == 'principal'")
    assert set(principal_rows["component"]) == {"observed", "pls_captured", "residual"}
    assert principal_rows["valid_replicates"].max() == 12
    # The replicate axis is anchored by its own net displacement, so a resampled
    # trajectory that still progresses the same way keeps its feature signs.
    observed_rows = principal_rows.query("component == 'observed'")
    assert observed_rows["sign_stability"].max() == pytest.approx(1.0)


def test_principal_bootstrap_counts_closed_replicates_as_unavailable() -> None:
    X, metadata = _inputs()
    # Both endpoint cells are constant, so every resample of group A returns to
    # its start and no replicate defines a principal orientation.
    for stage in ("t0", "t2"):
        X.loc[(metadata["group"] == "A") & (metadata["stage"] == stage), :] = [4.0, 4.0, 4.0]
    result = analyze_orientation_attribution(
        X, metadata, _model(X, metadata), bootstrap_replicates=8, bootstrap_seed=3
    )
    principal_rows = result.bootstrap_summaries.query(
        "transition_id == 'principal' and component == 'observed'"
    )
    assert principal_rows["valid_replicates"].max() == 0
    assert principal_rows["sign_stability"].isna().all()


def test_principal_frames_and_interpretation_state_the_decomposition(tmp_path) -> None:
    X, metadata = _inputs()
    result = analyze_orientation_attribution(X, metadata, _model(X, metadata))
    frames = result.frames()

    summary = frames["principal_orientation"]
    assert list(summary["component"]) == ["observed", "pls_captured", "residual"]
    assert set(summary["transition_id"]) == {"principal"}
    assert {
        "relative_eigengap_observed_group_1",
        "relative_eigengap_reconstructed_group_2",
        "eigengap_threshold",
        "degenerate",
        "degeneracy_sources",
    } <= set(summary.columns)
    vectors = frames["principal_orientation_vectors"]
    assert len(vectors) == 3 * 2 * 3
    assert {"principal_axis", "net_displacement"} <= set(vectors.columns)

    # Existing frame schemas are untouched.
    assert set(frames["transition_summaries"]["transition_id"]) == {"t0->t1", "t1->t2"}
    assert len(frames["transition_vectors"]) == 2 * 3 * 2 * 3

    interpretation = dict(zip(frames["interpretation"]["field"], frames["interpretation"]["value"]))
    assert "decomposition_boundary" in interpretation
    assert "principal" in interpretation["decomposition_boundary"]
    assert "two-stage" in interpretation["decomposition_boundary"]
    assert "angle" in result.interpretation.decomposition_boundary

    paths = write_attribution_outputs(result, tmp_path / "attribution")
    assert paths["principal_orientation"].exists()
    assert paths["principal_orientation_vectors"].exists()
