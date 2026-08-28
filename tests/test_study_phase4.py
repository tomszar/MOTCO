"""Phase 4 summaries, localization, and gate aggregation on synthetic records."""

from __future__ import annotations

import math

import pytest

from motco.simulations.grid import SimulationReplicateResult, summarize_rejection_rates
from motco.simulations.study.config import GateRule, Phase4GateConfig
from motco.simulations.study.phase4 import (
    Phase4SummaryError,
    build_operating_frame,
    evaluate_completeness,
    evaluate_phase4_gate,
    localize_off_diagonal,
    summarize_attribution,
    summarize_pls_selection,
    summarize_realized_geometry,
    type_i_inflation_bound,
)

_ALPHA = 0.05
_N = 100


def _gate(**overrides) -> Phase4GateConfig:
    defaults: dict = {
        "enabled": True,
        "alpha": _ALPHA,
        "control_se_tolerance": 2.0,
        "monotonicity_se_tolerance": 2.0,
        "min_power_at_top": 0.8,
        "confirmation_se_threshold": 1.0,
        "max_marginal_exceedances": 1,
        "control_modes": ("none", "translation"),
        "rules": (
            GateRule(trajectory_mode="magnitude", statistic="delta", role="mandatory_power"),
            GateRule(trajectory_mode="orientation", statistic="angle", role="mandatory_power"),
            GateRule(trajectory_mode="shape", statistic="shape", role="mandatory_power"),
            GateRule(trajectory_mode="magnitude", statistic="angle", role="mandatory_control"),
            GateRule(trajectory_mode="magnitude", statistic="shape", role="mandatory_control"),
            GateRule(trajectory_mode="orientation", statistic="delta", role="descriptive"),
            GateRule(trajectory_mode="shape", statistic="delta", role="descriptive"),
        ),
        "require_complete_records": True,
    }
    defaults.update(overrides)
    return Phase4GateConfig(**defaults)


def _record(
    cell_id: str,
    phase: str,
    replicate_index: int,
    *,
    metadata: dict,
    p_values: dict[str, float | None],
    geometry: dict | None = None,
    integration: dict | None = None,
    attribution_status: str = "not_requested",
    attribution: dict | None = None,
    status: str = "completed",
    diagnostic_error_message: str | None = None,
) -> SimulationReplicateResult:
    return SimulationReplicateResult(
        cell_id=cell_id,
        phase=phase,
        replicate_index=replicate_index,
        replicate_seed=replicate_index,
        generator_seed=replicate_index,
        evaluation_seed=replicate_index,
        parameter_signature="sig",
        status=status,  # type: ignore[arg-type]
        p_values=p_values,
        pair_statistics={"delta": 1.0, "angle": 0.5, "shape": 0.25},
        realized_geometry=geometry if geometry is not None else _geometry(),
        truth_metadata={},
        runtime_metadata={"runtime_seconds": 0.1},
        cell_metadata=metadata,
        integration_metadata=integration if integration is not None else _integration(),
        attribution_status=attribution_status,
        attribution_diagnostics=attribution or {},
        diagnostic_error_message=diagnostic_error_message,
    )


def _integration(selected_lv: int = 5) -> dict:
    return {
        "integration_method": "pls",
        "integration_metadata_version": 1,
        "selected_lv": selected_lv,
        "cv_mean_auroc": 0.9,
        "integration_params": {
            "cv1_splits": 3,
            "cv2_splits": 4,
            "n_repeats": 5,
            "max_components": 20,
            "random_state": 1203,
        },
    }


def _geometry(*, delta: float = 1.0, angle: float = 5.0, shape: float | None = 0.05) -> dict:
    def scope(d: float, a: float | None, sh: float | None) -> dict:
        return {
            "path_lengths": {"A": 3.0, "B": 3.0},
            "delta": d,
            "angle": a,
            "shape": sh,
            "availability": {"delta": True, "angle": a is not None, "shape": sh is not None},
        }

    return {
        "schema_version": 1,
        "requested": {"trajectory_mode": "magnitude", "group_effect_size": 1.0},
        "checkpoints": {
            "population_standardized": {"joint": scope(delta, angle, shape)},
            "observed_standardized": {"joint": scope(delta, angle, shape)},
            "pls_latent": {"joint": scope(delta * 2, angle, shape)},
        },
    }


def _cells(
    *,
    rates: dict[tuple[str, float], dict[str, float]],
    anchor_rate: dict[str, float] | None = None,
    modes: tuple[str, ...] = ("magnitude", "orientation", "shape", "translation"),
    control_rate: dict[str, float] | None = None,
) -> list[SimulationReplicateResult]:
    """Build records whose empirical rejection rates match ``rates`` exactly."""

    records: list[SimulationReplicateResult] = []

    def emit(cell_id: str, phase: str, metadata: dict, rate: dict[str, float]) -> None:
        for index in range(_N):
            p_values = {
                statistic: (0.001 if index < round(value * _N) else 0.9)
                for statistic, value in rate.items()
            }
            records.append(_record(cell_id, phase, index, metadata=metadata, p_values=p_values))

    emit(
        "type_i-none",
        "type_i_baseline",
        {"varied_axis": None, "varied_value": None},
        control_rate or {"delta": 0.04, "angle": 0.04, "shape": 0.04},
    )
    if anchor_rate is not None:
        emit(
            "power-anchor",
            "power_primary",
            {
                "trajectory_mode": "none",
                "effect_size": 0.0,
                "varied_axis": None,
                "zero_effect_anchor": True,
                "resolves_modes": list(modes),
            },
            anchor_rate,
        )
    for (mode, effect), rate in rates.items():
        emit(
            f"power-{mode}-{effect}",
            "power_primary",
            {"trajectory_mode": mode, "effect_size": effect, "varied_axis": None},
            rate,
        )
    return records


_PASSING_RATES: dict[tuple[str, float], dict[str, float]] = {
    ("magnitude", 0.5): {"delta": 0.5, "angle": 0.03, "shape": 0.03},
    ("magnitude", 1.0): {"delta": 0.95, "angle": 0.04, "shape": 0.04},
    ("orientation", 0.5): {"delta": 0.3, "angle": 0.6, "shape": 0.2},
    ("orientation", 1.0): {"delta": 0.5, "angle": 0.95, "shape": 0.4},
    ("shape", 0.5): {"delta": 0.2, "angle": 0.3, "shape": 0.6},
    ("shape", 1.0): {"delta": 0.3, "angle": 0.4, "shape": 0.9},
    ("translation", 0.5): {"delta": 0.03, "angle": 0.03, "shape": 0.03},
    ("translation", 1.0): {"delta": 0.04, "angle": 0.04, "shape": 0.04},
}


def _decide(records, gate: Phase4GateConfig | None = None, **kwargs):
    summaries = summarize_rejection_rates(records, alpha=_ALPHA)
    return evaluate_phase4_gate(gate or _gate(), summaries, records, **kwargs)


# --- gate outcomes ------------------------------------------------------------


def test_all_gates_met_emits_proceed() -> None:
    records = _cells(rates=_PASSING_RATES, anchor_rate={"delta": 0.03, "angle": 0.03, "shape": 0.03})
    decision = _decide(records)
    assert decision.decision == "proceed", decision.rationale
    assert decision.confirmation_runs == ()
    frame = decision.to_frame()
    assert not frame.empty
    assert set(frame["kind"]) >= {"type_i_inflation", "power", "control", "descriptive", "completeness"}


def test_single_marginal_control_exceedance_is_indeterminate() -> None:
    bound = type_i_inflation_bound(_ALPHA, 2.0, _N)
    # One control statistic just over the bound, by well under one MC SE.
    marginal = math.ceil((bound + 0.005) * _N) / _N
    records = _cells(
        rates=_PASSING_RATES,
        anchor_rate={"delta": 0.03, "angle": 0.03, "shape": 0.03},
        control_rate={"delta": marginal, "angle": 0.04, "shape": 0.04},
    )
    decision = _decide(records)
    assert decision.decision == "indeterminate", decision.rationale
    assert len(decision.confirmation_runs) == 1
    run = decision.confirmation_runs[0]
    assert run["cell_id"] == "type_i-none"
    assert run["statistic"] == "delta"
    assert "re-run" in run["action"]


def test_material_control_exceedance_holds() -> None:
    records = _cells(
        rates=_PASSING_RATES,
        anchor_rate={"delta": 0.03, "angle": 0.03, "shape": 0.03},
        control_rate={"delta": 0.25, "angle": 0.04, "shape": 0.04},
    )
    decision = _decide(records)
    assert decision.decision == "hold"
    assert "type_i_inflation[type_i-none,delta]" in decision.rationale


def test_two_marginal_exceedances_hold() -> None:
    bound = type_i_inflation_bound(_ALPHA, 2.0, _N)
    marginal = math.ceil((bound + 0.005) * _N) / _N
    records = _cells(
        rates=_PASSING_RATES,
        anchor_rate={"delta": 0.03, "angle": 0.03, "shape": 0.03},
        control_rate={"delta": marginal, "angle": marginal, "shape": 0.04},
    )
    decision = _decide(records)
    assert decision.decision == "hold"
    assert "marginal exceedance" in decision.rationale


def test_power_floor_failure_holds() -> None:
    rates = dict(_PASSING_RATES)
    rates[("shape", 1.0)] = {"delta": 0.3, "angle": 0.4, "shape": 0.4}
    records = _cells(rates=rates, anchor_rate={"delta": 0.03, "angle": 0.03, "shape": 0.03})
    decision = _decide(records)
    assert decision.decision == "hold"
    assert "mandatory_power[shape,shape]" in decision.rationale


def test_magnitude_off_diagonal_control_failure_holds() -> None:
    rates = dict(_PASSING_RATES)
    rates[("magnitude", 1.0)] = {"delta": 0.95, "angle": 0.5, "shape": 0.04}
    records = _cells(rates=rates, anchor_rate={"delta": 0.03, "angle": 0.03, "shape": 0.03})
    decision = _decide(records)
    assert decision.decision == "hold"
    assert "mandatory_control[magnitude,angle]" in decision.rationale


def test_mixed_construction_off_diagonals_are_descriptive_only() -> None:
    """Orientation's delta and shape's delta move a lot and must not gate."""

    rates = dict(_PASSING_RATES)
    rates[("orientation", 1.0)] = {"delta": 0.99, "angle": 0.95, "shape": 0.99}
    rates[("shape", 1.0)] = {"delta": 0.99, "angle": 0.99, "shape": 0.9}
    records = _cells(rates=rates, anchor_rate={"delta": 0.03, "angle": 0.03, "shape": 0.03})
    decision = _decide(records)
    assert decision.decision == "proceed", decision.rationale
    descriptive = [o for o in decision.observations if o.kind == "descriptive"]
    assert descriptive and all(o.met is None for o in descriptive)


# --- monotonicity -------------------------------------------------------------


def test_small_power_reversal_is_tolerated() -> None:
    rates = dict(_PASSING_RATES)
    # 0.5 -> 1.0 dips by 0.01, far inside two combined SEs at n=100.
    rates[("magnitude", 0.5)] = {"delta": 0.96, "angle": 0.03, "shape": 0.03}
    rates[("magnitude", 1.0)] = {"delta": 0.95, "angle": 0.04, "shape": 0.04}
    records = _cells(rates=rates, anchor_rate={"delta": 0.03, "angle": 0.03, "shape": 0.03})
    decision = _decide(records)
    assert decision.decision == "proceed", decision.rationale
    power = next(o for o in decision.observations if o.rule == "mandatory_power[magnitude,delta]")
    assert power.observations["reversals"] == []


def test_material_power_reversal_holds() -> None:
    rates = dict(_PASSING_RATES)
    rates[("magnitude", 0.5)] = {"delta": 0.95, "angle": 0.03, "shape": 0.03}
    rates[("magnitude", 1.0)] = {"delta": 0.5, "angle": 0.04, "shape": 0.04}
    records = _cells(rates=rates, anchor_rate={"delta": 0.03, "angle": 0.03, "shape": 0.03})
    decision = _decide(records)
    assert decision.decision == "hold"
    power = next(o for o in decision.observations if o.rule == "mandatory_power[magnitude,delta]")
    assert power.observations["reversals"]


# --- unavailable metrics and incomplete work ----------------------------------


def test_unavailable_statistic_yields_indeterminate_not_proceed() -> None:
    records = _cells(rates=_PASSING_RATES, anchor_rate={"delta": 0.03, "angle": 0.03, "shape": 0.03})
    # Strip every shape p-value so the shape power gate has no evidence.
    stripped = [
        r.__class__(**{**r.__dict__, "p_values": {k: v for k, v in r.p_values.items() if k != "shape"}})
        for r in records
    ]
    decision = _decide(stripped)
    assert decision.decision == "indeterminate"
    assert "mandatory_power[shape,shape]" in decision.rationale


def test_incomplete_work_units_block_proceed() -> None:
    records = _cells(rates=_PASSING_RATES, anchor_rate={"delta": 0.03, "angle": 0.03, "shape": 0.03})
    decision = _decide(records, expected_units=len(records) + 10)
    assert decision.decision == "hold"
    assert "expected work units" in decision.rationale


def test_missing_integration_metadata_blocks_proceed() -> None:
    records = _cells(rates=_PASSING_RATES, anchor_rate={"delta": 0.03, "angle": 0.03, "shape": 0.03})
    broken = [records[0].__class__(**{**records[0].__dict__, "integration_metadata": {}}), *records[1:]]
    decision = _decide(broken)
    assert decision.decision == "hold"
    assert "selected-component metadata" in decision.rationale


def test_eligible_attribution_without_result_blocks_proceed() -> None:
    records = _cells(rates=_PASSING_RATES, anchor_rate={"delta": 0.03, "angle": 0.03, "shape": 0.03})
    broken = [
        records[0].__class__(
            **{**records[0].__dict__, "attribution_status": "computed", "attribution_diagnostics": {}}
        ),
        *records[1:],
    ]
    decision = _decide(broken)
    assert decision.decision == "hold"
    assert "attribution diagnostics" in decision.rationale


def test_recorded_attribution_failure_is_acceptable() -> None:
    records = _cells(rates=_PASSING_RATES, anchor_rate={"delta": 0.03, "angle": 0.03, "shape": 0.03})
    with_failure = [
        records[0].__class__(
            **{
                **records[0].__dict__,
                "attribution_status": "failed",
                "attribution_diagnostics": {"status": "failed", "reason": "no direction"},
                "diagnostic_error_message": "no direction",
            }
        ),
        *records[1:],
    ]
    decision = _decide(with_failure)
    assert decision.decision == "proceed", decision.rationale


def test_gate_requires_enabled_configuration() -> None:
    records = _cells(rates=_PASSING_RATES)
    with pytest.raises(Phase4SummaryError, match="acceptance.gate.enabled"):
        _decide(records, gate=Phase4GateConfig())


# --- anchor expansion ---------------------------------------------------------


def test_zero_effect_anchor_supplies_every_mode_curve() -> None:
    records = _cells(rates=_PASSING_RATES, anchor_rate={"delta": 0.03, "angle": 0.03, "shape": 0.03})
    frame = build_operating_frame(summarize_rejection_rates(records, alpha=_ALPHA), records)
    for mode in ("magnitude", "orientation", "shape", "translation"):
        zero = frame[(frame["trajectory_mode"] == mode) & (frame["effect_size"] == 0.0)]
        assert len(zero) == 3, mode
        assert zero["from_shared_anchor"].all()
        assert set(zero["cell_id"]) == {"power-anchor"}


# --- descriptive summaries ----------------------------------------------------


def test_geometry_summary_preserves_unavailable_and_labels_spaces() -> None:
    records = [
        _record(
            "power-magnitude-1.0",
            "power_primary",
            index,
            metadata={"trajectory_mode": "magnitude", "effect_size": 1.0, "varied_axis": None},
            p_values={"delta": 0.01},
            geometry=_geometry(shape=None if index % 2 else 0.05),
        )
        for index in range(4)
    ]
    frame = summarize_realized_geometry(records)
    shape_rows = frame[(frame["statistic"] == "shape") & (frame["checkpoint"] == "pls_latent")]
    assert len(shape_rows) == 1
    assert int(shape_rows.iloc[0]["n_available"]) == 2
    assert int(shape_rows.iloc[0]["n_unavailable"]) == 2

    spaces = dict(zip(frame["checkpoint"], frame["measurement_space"]))
    assert spaces["observed_standardized"] == "standardized_feature"
    assert spaces["pls_latent"] == "pls_latent"
    assert "path_length[A]" in set(frame["statistic"])


def test_pls_selection_summary_reports_components_and_missing() -> None:
    records = [
        _record(
            "power-magnitude-1.0",
            "power_primary",
            index,
            metadata={"trajectory_mode": "magnitude", "effect_size": 1.0},
            p_values={"delta": 0.01},
            integration=_integration(selected_lv=4 + (index % 3)) if index < 3 else {},
        )
        for index in range(4)
    ]
    frame = summarize_pls_selection(records)
    row = frame.iloc[0]
    assert row["n_selected_lv"] == 3
    assert row["selected_lv_min"] == 4
    assert row["selected_lv_max"] == 6
    assert row["missing_integration_metadata"] == 1
    assert row["cv_settings_consistent"]
    assert "cv1_splits=3" in row["cv_settings"]


def _attribution_payload(features: list[str], *, signs: list[int] | None = None) -> dict:
    signs = signs or [1] * len(features)
    return {
        "schema_version": 1,
        "status": "computed",
        "transitions": [
            {
                "transition_id": "0->1",
                "retention": {"available": True, "cosine": 0.8, "norm_ratio": 0.7, "residual_fraction": 0.3},
                "components": {},
            }
        ],
        "top_features": [
            {"transition_id": "0->1", "component": "observed", "rank": i + 1, "feature": name, "sign": sign}
            for i, (name, sign) in enumerate(zip(features, signs))
        ],
        "truth_recovery": [
            {
                "transition_id": "0->1",
                "component": "observed",
                "available": True,
                "precision": 0.5,
                "recall": 0.1,
                "selected_count": len(features),
            }
        ],
        "stability": [
            {
                "transition_id": "0->1",
                "component": component,
                "mean_sign_stability": 0.9,
                "mean_top_k_selection_frequency": 0.6,
            }
            for component in ("observed", "pls_captured", "residual")
        ],
    }


def test_attribution_summary_reports_agreement_and_availability() -> None:
    payloads = [
        _attribution_payload(["f1", "f2", "f3"]),
        _attribution_payload(["f1", "f2", "f4"]),
        _attribution_payload(["f1", "f2", "f5"], signs=[1, -1, 1]),
    ]
    records = [
        _record(
            "power-orientation-1.0",
            "power_primary",
            index,
            metadata={"trajectory_mode": "orientation", "effect_size": 1.0},
            p_values={"angle": 0.01},
            attribution_status="computed",
            attribution=payload,
        )
        for index, payload in enumerate(payloads)
    ]
    frame = summarize_attribution(records)
    observed = frame[frame["component"] == "observed"].iloc[0]
    assert observed["eligible_replicates"] == 3
    assert observed["computed_replicates"] == 3
    assert observed["availability_rate"] == pytest.approx(1.0)
    assert observed["retention_cosine_mean"] == pytest.approx(0.8)
    assert observed["precision_mean"] == pytest.approx(0.5)
    assert observed["bootstrap_sign_stability_mean"] == pytest.approx(0.9)
    # f1/f2 shared by all three; each pair overlaps on 2 of 4 → Jaccard 0.5.
    assert observed["top_k_jaccard"] == pytest.approx(0.5)
    # f1 unanimous (1.0), f2 two-of-three on the modal sign (2/3).
    assert observed["sign_agreement"] == pytest.approx((1.0 + 2 / 3) / 2)


def test_attribution_summary_skips_cells_that_were_not_requested() -> None:
    records = _cells(rates=_PASSING_RATES)
    assert summarize_attribution(records).empty


def test_localization_never_compares_raw_distances_across_spaces() -> None:
    """delta is normalized by each checkpoint's own path length, not compared raw."""

    from motco.simulations.study.phase4 import _normalized_geometry, summarize_realized_geometry

    records = [
        _record(
            "power-orientation-1.0",
            "power_primary",
            index,
            metadata={"trajectory_mode": "orientation", "effect_size": 1.0},
            p_values={"angle": 0.01},
            geometry=_geometry(delta=1.5, angle=72.0, shape=0.05),
        )
        for index in range(2)
    ]
    scoped = summarize_realized_geometry(records)
    scoped = scoped[scoped["scope"] == "joint"]
    normalized = _normalized_geometry(scoped)
    # Feature-space delta 1.5 over a path length of 3.0; latent delta is 3.0 over
    # the same path length. Both become dimensionless before any comparison.
    assert normalized[("power-orientation-1.0", "observed_standardized", "delta")] == pytest.approx(0.5)
    assert normalized[("power-orientation-1.0", "pls_latent", "delta")] == pytest.approx(1.0)
    assert normalized[("power-orientation-1.0", "pls_latent", "angle")] == pytest.approx(72.0 / 180)


def test_localization_names_the_first_material_checkpoint() -> None:
    anchor = [
        _record(
            "power-anchor",
            "power_primary",
            index,
            metadata={
                "trajectory_mode": "none",
                "effect_size": 0.0,
                "zero_effect_anchor": True,
                "resolves_modes": ["magnitude"],
            },
            p_values={"delta": 0.5},
            geometry=_geometry(delta=1.0, angle=5.0, shape=0.05),
        )
        for index in range(4)
    ]
    # Magnitude's off-diagonal angle is already 5x the null at the population
    # checkpoint: construction-present, not a projection artifact.
    magnitude = [
        _record(
            "power-magnitude-1.0",
            "power_primary",
            index,
            metadata={"trajectory_mode": "magnitude", "effect_size": 1.0},
            p_values={"delta": 0.01},
            geometry=_geometry(delta=4.0, angle=40.0, shape=0.05),
        )
        for index in range(4)
    ]
    frame = localize_off_diagonal(anchor + magnitude)
    angle_row = frame[(frame["trajectory_mode"] == "magnitude") & (frame["statistic"] == "angle")].iloc[0]
    assert angle_row["first_material_checkpoint"] == "population_standardized"
    assert angle_row["classification"] == "construction_present"
    assert angle_row["measurement_space"] == "standardized_feature"
    # angle is reported in degrees and normalized by its 180-degree maximum:
    # 40 deg against a 5 deg null.
    assert angle_row["normalized_value"] == pytest.approx(40.0 / 180)
    assert angle_row["normalized_null"] == pytest.approx(5.0 / 180)
    assert angle_row["normalized_excess"] == pytest.approx(35.0 / 180)

    # shape is identical to the null at every checkpoint, so it stays immaterial.
    shape_row = frame[(frame["trajectory_mode"] == "magnitude") & (frame["statistic"] == "shape")].iloc[0]
    assert shape_row["first_material_checkpoint"] is None
    assert shape_row["classification"] == "not_material"

    # The mode's own diagonal statistic is never reported as off-diagonal.
    assert "delta" not in set(frame[frame["trajectory_mode"] == "magnitude"]["statistic"])


def test_completeness_counts_failed_units() -> None:
    ok = _record(
        "cell",
        "power_primary",
        0,
        metadata={"trajectory_mode": "magnitude", "effect_size": 1.0},
        p_values={"delta": 0.01},
    )
    failed = _record(
        "cell",
        "power_primary",
        1,
        metadata={"trajectory_mode": "magnitude", "effect_size": 1.0},
        p_values={},
        status="failed",
    )
    observation = evaluate_completeness([ok, failed])
    assert observation.met is False
    assert observation.observations["failed"] == 1
