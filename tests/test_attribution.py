from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from motco.stats import (
    AttributionError,
    analyze_orientation_attribution,
    attribution_frames,
    fit_plsda_model,
    write_attribution_outputs,
)


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
    assert len(result.aggregate_effects) == 2 * 3 * 2
    frames = attribution_frames(result)
    assert {"feature_effects", "transition_summaries", "aggregate_effects", "configuration", "interpretation"} <= set(
        frames
    )
    paths = write_attribution_outputs(result, tmp_path / "attribution")
    assert (tmp_path / "attribution" / "feature_effects.csv").exists()
    assert paths["interpretation"].read_text(encoding="utf-8").find("causal_boundary") >= 0
