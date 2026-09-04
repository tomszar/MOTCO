"""Deterministic unit tests for the simulation attribution adapter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from motco.simulations.attribution_diagnostics import (
    ATTRIBUTION_SCHEMA_VERSION,
    AttributionDiagnosticError,
    compute_attribution_diagnostics,
    derive_truth_driver_features,
    flatten_attribution_diagnostics,
    unavailable_record,
)
from motco.simulations.evaluation import AttributionDiagnosticSettings
from motco.simulations.semisynthetic import SemiSyntheticTrajectoryDataset
from motco.stats.pls import fit_plsda_model

_N_METHYL = 4
_N_EXPR = 3
_N_PROT = 2
_N_FEATURES = _N_METHYL + _N_EXPR + _N_PROT


def _feature_names() -> list[str]:
    return (
        [f"methylation__cg{i}" for i in range(_N_METHYL)]
        + [f"expression__gene{i}" for i in range(_N_EXPR)]
        + [f"proteomics__prot{i}" for i in range(_N_PROT)]
    )


def _indicators(pattern: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {layer: np.asarray(values, dtype=float) for layer, values in pattern.items()}


def _dataset(
    *,
    n_stages: int = 3,
    per_group: int = 4,
    indicators_a: dict[str, np.ndarray] | None = None,
    indicators_b: dict[str, np.ndarray] | None = None,
    deltas: dict[str, list[float]] | None = None,
    truth: dict | None = None,
    seed: int = 0,
) -> tuple[SemiSyntheticTrajectoryDataset, pd.DataFrame]:
    """Build a tiny aligned dataset plus its standardized joint feature matrix."""

    rng = np.random.default_rng(seed)
    rows = [
        (group, str(stage))
        for group in ("A", "B")
        for stage in range(n_stages)
        for _ in range(per_group)
    ]
    metadata = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(len(rows))],
            "group": [group for group, _ in rows],
            "stage": [stage for _, stage in rows],
        }
    )
    names = _feature_names()
    values = rng.normal(size=(len(rows), _N_FEATURES))
    # Give each stage a reproducible offset so the PLS fit is not degenerate.
    stage_index = metadata["stage"].astype(int).to_numpy()
    values += stage_index[:, None] * np.linspace(0.5, 1.5, _N_FEATURES)[None, :]
    features = pd.DataFrame(values, columns=names)

    default_truth: dict = {
        "group_labels": ["A", "B"],
        "deltas": deltas or {"A": [2.0, 2.0, 2.0], "B": [2.0, 2.0, 2.0]},
        "indicators": {
            "A": indicators_a or _zero_indicators(n_stages),
            "B": indicators_b or _zero_indicators(n_stages),
        },
    }
    if truth is not None:
        default_truth = truth

    dataset = SemiSyntheticTrajectoryDataset(
        methylation=pd.DataFrame(values[:, :_N_METHYL], columns=[f"cg{i}" for i in range(_N_METHYL)]),
        expression=pd.DataFrame(
            values[:, _N_METHYL : _N_METHYL + _N_EXPR], columns=[f"gene{i}" for i in range(_N_EXPR)]
        ),
        proteomics=pd.DataFrame(values[:, _N_METHYL + _N_EXPR :], columns=[f"prot{i}" for i in range(_N_PROT)]),
        metadata=metadata,
        truth=default_truth,
    )
    return dataset, features


def _zero_indicators(n_stages: int) -> dict[str, np.ndarray]:
    return _indicators(
        {
            "methylation": np.zeros((_N_METHYL, n_stages)),
            "expression": np.zeros((_N_EXPR, n_stages)),
            "proteomics": np.zeros((_N_PROT, n_stages)),
        }
    )


def _run(dataset, features, *, settings: AttributionDiagnosticSettings, components: int = 2) -> dict:
    model = fit_plsda_model(features, dataset.metadata["stage"], n_components=components)
    return compute_attribution_diagnostics(
        dataset,
        model=model,
        features=features,
        original_scales=np.full(_N_FEATURES, 2.0),
        settings=settings,
        group_col="group",
        stage_col="stage",
        selected_components=components,
        feature_order_signature="deadbeef",
    )


# --- multi-stage extraction ---------------------------------------------------


@pytest.mark.parametrize("n_stages", [3, 4, 5])
def test_extraction_covers_every_adjacent_transition(n_stages: int) -> None:
    dataset, features = _dataset(n_stages=n_stages)
    record = _run(dataset, features, settings=AttributionDiagnosticSettings(enabled=True, top_k=3))

    expected = [f"{i}->{i + 1}" for i in range(n_stages - 1)]
    assert [entry["transition_id"] for entry in record["transitions"]] == expected
    for entry in record["transitions"]:
        assert entry["from_stage"] != entry["to_stage"]
        assert set(entry["components"]) == {"observed", "pls_captured", "residual"}
        assert entry["components"]["observed"]["path_length_group_1"] is not None
        assert entry["retention"]["available"] is True
        assert -1.0 - 1e-9 <= entry["retention"]["cosine"] <= 1.0 + 1e-9
    assert record["schema_version"] == ATTRIBUTION_SCHEMA_VERSION


# --- propagated truth recovery ------------------------------------------------


def test_truth_includes_propagated_expression_and_protein_drivers() -> None:
    n_stages = 3
    ind_a = _zero_indicators(n_stages)
    ind_b = _zero_indicators(n_stages)
    # A methylation site turns on at stage 1 for group B only; the generator's
    # cascade turns the mapped gene and protein on with it.
    ind_b["methylation"][0, 1] = 1.0
    ind_b["expression"][0, 1] = 1.0
    ind_b["proteomics"][0, 1] = 1.0

    dataset, _ = _dataset(n_stages=n_stages, indicators_a=ind_a, indicators_b=ind_b)
    truth = derive_truth_driver_features(dataset, _feature_names(), ["0", "1", "2"])

    assert truth.available is True
    assert set(truth.names()) == {"methylation__cg0", "expression__gene0", "proteomics__prot0"}
    # The site switches on across 0->1 and back off across 1->2.
    assert set(truth.names("0->1")) == set(truth.names("1->2"))


def test_truth_covers_effect_size_only_constructions() -> None:
    """`magnitude_kind='all'` leaves indicators identical and scales delta."""

    n_stages = 3
    ind = _zero_indicators(n_stages)
    ind["methylation"][2, 1] = 1.0
    dataset, _ = _dataset(
        n_stages=n_stages,
        indicators_a=ind,
        indicators_b={k: v.copy() for k, v in ind.items()},
        deltas={"A": [2.0, 2.0, 2.0], "B": [3.0, 2.0, 2.0]},
    )
    truth = derive_truth_driver_features(dataset, _feature_names(), ["0", "1", "2"])
    assert truth.names() == ["methylation__cg2"]


def test_identical_groups_have_no_truth_drivers() -> None:
    dataset, _ = _dataset()
    truth = derive_truth_driver_features(dataset, _feature_names(), ["0", "1", "2"])
    assert truth.available is True
    assert truth.names() == []


def test_truth_marked_unavailable_without_indicators() -> None:
    dataset, _ = _dataset(truth={"trajectory_mode": "none"})
    truth = derive_truth_driver_features(dataset, _feature_names(), ["0", "1", "2"])
    assert truth.available is False
    assert truth.names() == []
    assert "unavailable" in truth.definition


def test_misaligned_truth_indicators_are_rejected() -> None:
    dataset, _ = _dataset()
    with pytest.raises(AttributionDiagnosticError, match="do not align"):
        derive_truth_driver_features(dataset, _feature_names()[:-1], ["0", "1", "2"])


# --- unavailable transitions --------------------------------------------------


def test_unavailable_transition_is_marked_not_zero() -> None:
    """A group with no movement over a transition has no direction to report."""

    dataset, features = _dataset(n_stages=3, per_group=4, seed=3)
    # Collapse group B's stage 0 and stage 1 onto identical feature values so its
    # transition vector is exactly zero.
    frame = features.copy()
    meta = dataset.metadata
    stage0 = np.flatnonzero((meta["group"] == "B") & (meta["stage"] == "0"))
    stage1 = np.flatnonzero((meta["group"] == "B") & (meta["stage"] == "1"))
    frame.iloc[stage1, :] = frame.iloc[stage0, :].to_numpy()

    record = _run(dataset, frame, settings=AttributionDiagnosticSettings(enabled=True, top_k=2))
    first = record["transitions"][0]
    assert first["components"]["observed"]["contrast_available"] is False
    assert first["components"]["observed"]["contrast_norm"] is None
    assert first["retention"]["available"] is False
    assert first["retention"]["cosine"] is None
    assert first["retention"]["norm_ratio"] is None
    # Path length is still reported; only the direction is unavailable.
    assert first["components"]["observed"]["path_length_group_2"] == pytest.approx(0.0)

    unavailable = [
        entry
        for entry in record["truth_recovery"]
        if entry["transition_id"] == first["transition_id"] and entry["component"] == "observed"
    ]
    assert unavailable and unavailable[0]["available"] is False
    assert unavailable[0]["precision"] is None


# --- top-k truncation ---------------------------------------------------------


@pytest.mark.parametrize("top_k", [1, 3, _N_FEATURES + 5])
def test_top_k_truncates_and_never_exceeds_the_feature_count(top_k: int) -> None:
    dataset, features = _dataset()
    record = _run(dataset, features, settings=AttributionDiagnosticSettings(enabled=True, top_k=top_k))

    counts: dict[tuple[str, str], int] = {}
    for entry in record["top_features"]:
        counts[(entry["transition_id"], entry["component"])] = (
            counts.get((entry["transition_id"], entry["component"]), 0) + 1
        )
    assert counts
    assert max(counts.values()) == min(top_k, _N_FEATURES)
    for entry in record["top_features"]:
        assert entry["effect_original"] == pytest.approx(entry["effect_standardized"] * 2.0)
    ranks = [entry["rank"] for entry in record["top_features"] if entry["component"] == "observed"]
    assert ranks[: min(top_k, _N_FEATURES)] == list(range(1, min(top_k, _N_FEATURES) + 1))


def test_top_features_are_ordered_by_absolute_effect() -> None:
    dataset, features = _dataset()
    record = _run(dataset, features, settings=AttributionDiagnosticSettings(enabled=True, top_k=_N_FEATURES))
    observed = [
        entry
        for entry in record["top_features"]
        if entry["component"] == "observed" and entry["transition_id"] == "0->1"
    ]
    magnitudes = [abs(entry["effect_standardized"]) for entry in observed]
    assert magnitudes == sorted(magnitudes, reverse=True)


# --- bootstrap reproducibility and disabled behavior --------------------------


def test_repeated_seeds_reproduce_bootstrap_stability() -> None:
    dataset, features = _dataset(per_group=6, seed=5)
    settings = AttributionDiagnosticSettings(enabled=True, bootstrap_replicates=8, bootstrap_seed=17, top_k=3)
    first = _run(dataset, features, settings=settings)
    second = _run(dataset, features, settings=settings)
    assert first["stability"] == second["stability"]
    assert first["top_features"] == second["top_features"]

    changed = _run(
        dataset,
        features,
        settings=AttributionDiagnosticSettings(
            enabled=True, bootstrap_replicates=8, bootstrap_seed=18, top_k=3
        ),
    )
    assert changed["stability"] != first["stability"]


def test_bootstrap_summaries_are_recorded_per_component() -> None:
    dataset, features = _dataset(per_group=6, seed=5)
    record = _run(
        dataset,
        features,
        settings=AttributionDiagnosticSettings(enabled=True, bootstrap_replicates=5, top_k=3),
    )
    keys = {(entry["transition_id"], entry["component"]) for entry in record["stability"]}
    assert keys == {
        (block, component)
        # the adjacent transitions plus the principal-orientation block
        for block in ("0->1", "1->2", "principal")
        for component in ("observed", "pls_captured", "residual")
    }
    for entry in record["stability"]:
        assert entry["requested_replicates"] == 5
        assert 0 <= entry["valid_replicates"] <= 5


def test_unavailable_record_distinguishes_not_requested_from_failed() -> None:
    not_requested = unavailable_record("not selected")
    failed = unavailable_record("boom", status="failed")
    assert not_requested["status"] == "not_requested"
    assert failed["status"] == "failed"
    assert failed["reason"] == "boom"
    assert not_requested["schema_version"] == ATTRIBUTION_SCHEMA_VERSION


def test_flatten_extracts_scalar_fields() -> None:
    dataset, features = _dataset()
    record = _run(dataset, features, settings=AttributionDiagnosticSettings(enabled=True, top_k=2))
    flat = flatten_attribution_diagnostics(record)
    assert flat["attribution_status"] == "computed"
    assert flat["attribution_top_k"] == 2
    assert flat["attribution_selected_components"] == 2
    assert flat["attribution_feature_order_signature"] == "deadbeef"
    assert flat["attribution_seconds"] >= 0.0


# --- principal-orientation block ----------------------------------------------


def test_principal_orientation_block_is_recorded_beside_the_transitions() -> None:
    """The record carries both decompositions, keyed by the reserved identifier."""

    dataset, features = _dataset(per_group=6, seed=5)
    record = _run(
        dataset,
        features,
        settings=AttributionDiagnosticSettings(enabled=True, bootstrap_replicates=4, top_k=3),
    )
    # The per-transition block is unchanged: `principal` is not a pseudo-transition.
    assert [entry["transition_id"] for entry in record["transitions"]] == ["0->1", "1->2"]

    for section in ("top_features", "truth_recovery", "stability"):
        assert any(entry["transition_id"] == "principal" for entry in record[section])

    # Its truth set is the union over transitions, so the row is scored rather
    # than left permanently unavailable.
    principal = [
        entry
        for entry in record["truth_recovery"]
        if entry["transition_id"] == "principal" and entry["component"] == "observed"
    ]
    assert len(principal) == 1
    assert principal[0]["available"] is True
    assert principal[0]["precision"] is not None
