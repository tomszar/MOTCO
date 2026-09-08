"""Retained-rank ladder reporting: operating column, decision rule, figure, wiring.

Records are built by hand with known p-values and known selected ranks so every
expected rate, threshold, and verdict is constructed rather than inferred.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from motco.simulations.grid import SimulationReplicateResult
from motco.simulations.study import (
    RankDecisionRule,
    StatisticPair,
    StudyReportError,
    TypeIBound,
    build_report_frames,
    evaluate_rank_decision,
    render_design_point_power,
    render_rank_ladder,
    resolve_operating_by_design_point,
    write_rank_decision,
)
from motco.simulations.study.enumerate import DESIGN_AXIS_MARKER, DESIGN_PHASE, DESIGN_POINT_KEY
from motco.simulations.study.report import assert_production_component_selection
from motco.simulations.study.spectrum import RANK_AXIS

RHO = "generator.baseline_continuity"
N = "generator.n_samples"


def record(
    *,
    rank: int | None,
    mode: str,
    effect: float,
    index: int,
    p: dict[str, float],
    selected_lv: int | None = None,
    declare: bool = True,
    point: dict | None = None,
) -> SimulationReplicateResult:
    """One record at rank ``rank`` (``None`` = the CV column, phase power_primary)."""

    is_cv = rank is None
    phase = "power_primary" if is_cv else DESIGN_PHASE
    coords = dict(point) if point is not None else {RANK_AXIS: rank}
    meta: dict = {"trajectory_mode": mode, "effect_size": effect, "varied_axis": None if is_cv else DESIGN_AXIS_MARKER}
    if declare:
        meta[DESIGN_POINT_KEY] = coords
    integration = {
        "integration_method": "pls",
        "component_selection": "cv" if is_cv else "forced",
        "selected_lv": selected_lv if selected_lv is not None else (3 if is_cv else rank),
    }
    return SimulationReplicateResult(
        cell_id=f"{phase}-{mode}-{effect}-rank{rank}",
        phase=phase,
        replicate_index=index,
        replicate_seed=index,
        generator_seed=index,
        evaluation_seed=index,
        parameter_signature="sig",
        status="completed",
        p_values=dict(p),
        cell_metadata=meta,
        integration_metadata=integration,
        null_summary={"angle": {"q95": 90.0}},
        config_spectrum={
            "version": 1,
            "pooled": {
                "n_points": 4,
                "n_dimensions": 3,
                "total_variance": 4.0,
                "spectrum": [],
                "relative_eigengap": 0.05,
            },
            "groups": {},
        },
    )


def _rates(n: int, rate: float) -> list[float]:
    k = int(round(rate * n))
    return [0.01] * k + [0.5] * (n - k)


def column(
    rank: int | None,
    *,
    n: int = 50,
    angle_power: float,
    delta_power: float = 1.0,
    shape_power: float = 1.0,
    anchor: dict[str, float] | None = None,
    selected_lv: int | None = None,
) -> list[SimulationReplicateResult]:
    """Anchor + magnitude/orientation/shape/translation at effects 0.5 and 1.0."""

    anchor = anchor or {}
    rows: list[SimulationReplicateResult] = []
    plans = {
        "orientation": {"angle": _rates(n, angle_power)},
        "magnitude": {"delta": _rates(n, delta_power)},
        "shape": {"shape": _rates(n, shape_power)},
        "translation": {},
    }
    anchor_p = {stat: _rates(n, anchor.get(stat, 0.0)) for stat in ("delta", "angle", "shape")}
    for index in range(n):
        rows.append(
            record(
                rank=rank,
                mode="none",
                effect=0.0,
                index=index,
                p={stat: anchor_p[stat][index] for stat in anchor_p},
                selected_lv=selected_lv,
            )
        )
        for mode, plan in plans.items():
            for effect in (0.5, 1.0):
                p = {"delta": 0.5, "angle": 0.5, "shape": 0.5}
                if effect == 1.0:
                    for stat, values in plan.items():
                        p[stat] = values[index]
                rows.append(record(rank=rank, mode=mode, effect=effect, index=index, p=p, selected_lv=selected_lv))
    return rows


def _rule(**overrides) -> RankDecisionRule:
    defaults: dict = {
        "axis": RANK_AXIS,
        "target": StatisticPair("orientation", "angle"),
        "protected": (StatisticPair("magnitude", "delta"), StatisticPair("shape", "shape")),
        "type_i_bound": TypeIBound(alpha=0.05, se_tolerance=2.0),
        "gain_se_multiplier": 2.0,
        "loss_se_multiplier": 2.0,
    }
    defaults.update(overrides)
    return RankDecisionRule(**defaults)


# ── 3.1 operating table ───────────────────────────────────────────────────────


def test_operating_table_marks_forced_rows_and_cv_rows() -> None:
    records = column(None, angle_power=0.5) + column(4, angle_power=0.6) + column(6, angle_power=0.9)
    frame = resolve_operating_by_design_point(records)
    assert RANK_AXIS in frame.columns and "component_selection" in frame.columns
    forced = frame[frame[RANK_AXIS].notna()]
    assert set(forced["component_selection"]) == {"forced"}
    assert (forced["median_selected_lv"] == forced[RANK_AXIS].astype(float)).all()
    cv = frame[frame[RANK_AXIS].isna()]
    assert set(cv["component_selection"]) == {"cv"}
    assert set(cv["is_baseline"]) == {True}
    assert set(cv["median_selected_lv"]) == {3.0}
    # Anchors are present per column as `none` at 0.0.
    anchors = frame[(frame["trajectory_mode"] == "none") & (frame["effect_size"] == 0.0)]
    assert len(anchors) == 3 * 3


def test_operating_table_is_unchanged_apart_from_the_new_column() -> None:
    from tests.test_study_design_point_report import _grid_records

    frame = resolve_operating_by_design_point(_grid_records())
    assert list(frame.columns)[-1] == "component_selection"
    assert set(frame["component_selection"]) == {"cv"}
    stripped = [r for r in _grid_records()]
    for r in stripped:
        r.integration_metadata.pop("component_selection", None)
    legacy = resolve_operating_by_design_point(stripped)
    assert legacy["component_selection"].isna().all()
    pd.testing.assert_frame_equal(
        frame.drop(columns=["component_selection"]), legacy.drop(columns=["component_selection"])
    )


def test_mixed_selection_in_one_column_is_flagged() -> None:
    records = column(None, angle_power=0.5) + column(4, angle_power=0.6)
    records[-1].integration_metadata["component_selection"] = "cv"
    frame = resolve_operating_by_design_point(records)
    assert "mixed" in set(frame[frame[RANK_AXIS] == 4]["component_selection"])


# ── 3.3 decision rule ─────────────────────────────────────────────────────────


def test_no_rank_qualifies_keeps_cv() -> None:
    records = column(None, angle_power=0.80) + column(4, angle_power=0.82) + column(6, angle_power=0.84)
    decision = evaluate_rank_decision(records, _rule())
    assert decision.verdict == "keep_cv"
    assert decision.chosen_rank is None
    assert decision.reference is not None and decision.reference.is_reference
    assert [c.rank for c in decision.columns] == [None, 4, 6]
    for c in decision.columns[1:]:
        assert c.qualifies is False
        gain = c.criterion("gain")
        assert gain is not None and gain.passed is False and gain.failing == ("orientation/angle",)
        assert c.criterion("anchor").passed is True
        assert c.criterion("protected").passed is True


def test_smallest_qualifying_rank_is_adopted() -> None:
    records = (
        column(None, angle_power=0.50)
        + column(4, angle_power=0.55)
        + column(6, angle_power=0.94)
        + column(9, angle_power=1.00)
    )
    decision = evaluate_rank_decision(records, _rule())
    assert decision.verdict == "adopt_fixed_rank"
    assert decision.chosen_rank == 6
    by_rank = {c.rank: c for c in decision.columns}
    assert by_rank[4].qualifies is False and by_rank[6].qualifies and by_rank[9].qualifies
    gain = by_rank[6].criterion("gain")
    assert gain is not None and gain.observations["difference"] == pytest.approx(0.44)
    assert gain.observations["threshold"] == pytest.approx(
        2.0 * ((0.94 * 0.06 / 50) + (0.5 * 0.5 / 50)) ** 0.5
    )


def test_protected_loss_vetoes_and_names_the_pair() -> None:
    records = column(None, angle_power=0.50) + column(6, angle_power=0.95, delta_power=0.60)
    decision = evaluate_rank_decision(records, _rule())
    assert decision.verdict == "keep_cv"
    col = decision.columns[1]
    assert col.criterion("gain").passed is True
    protected = col.criterion("protected")
    assert protected.passed is False and protected.failing == ("magnitude/delta",)
    assert protected.observations["magnitude_delta_loss"] == pytest.approx(0.40)
    assert "magnitude/delta" in protected.detail


def test_anchor_inflation_vetoes_and_names_the_statistic() -> None:
    records = column(None, angle_power=0.50) + column(9, angle_power=0.95, anchor={"shape": 0.30})
    decision = evaluate_rank_decision(records, _rule())
    assert decision.verdict == "keep_cv"
    col = decision.columns[1]
    anchor = col.criterion("anchor")
    assert anchor.passed is False and anchor.failing == ("shape",)
    assert anchor.observations["shape_rate"] == pytest.approx(0.30)
    assert anchor.observations["shape_bound"] == pytest.approx(0.05 + 2.0 * (0.05 * 0.95 / 50) ** 0.5)
    assert "shape" in anchor.detail


def test_thresholds_are_read_from_the_rule() -> None:
    records = column(None, angle_power=0.50) + column(6, angle_power=0.70, delta_power=0.90)
    lax = evaluate_rank_decision(records, _rule(gain_se_multiplier=0.5, loss_se_multiplier=3.0))
    strict = evaluate_rank_decision(records, _rule(gain_se_multiplier=2.5, loss_se_multiplier=0.5))
    assert lax.verdict == "adopt_fixed_rank" and lax.chosen_rank == 6
    assert strict.verdict == "keep_cv"
    assert strict.columns[1].criterion("gain").observations["se_multiplier"] == 2.5
    assert strict.columns[1].criterion("protected").observations["se_multiplier"] == 0.5
    tight_anchor = evaluate_rank_decision(
        column(None, angle_power=0.5) + column(6, angle_power=0.95, anchor={"angle": 0.06}),
        _rule(type_i_bound=TypeIBound(alpha=0.05, se_tolerance=0.0)),
    )
    assert tight_anchor.columns[1].criterion("anchor").failing == ("angle",)
    assert lax.rule["gain_se_multiplier"] == 0.5 and lax.rule["axis"] == RANK_AXIS


def test_decision_without_design_records_or_reference() -> None:
    assert evaluate_rank_decision(column(None, angle_power=0.5), _rule()).verdict == "no_design_grid"
    no_ref = evaluate_rank_decision(column(4, angle_power=0.9) + column(6, angle_power=0.9), _rule())
    assert no_ref.verdict == "no_reference" and [c.rank for c in no_ref.columns] == [4, 6]


def test_rank_decision_files_are_written(tmp_path: Path) -> None:
    records = column(None, angle_power=0.50) + column(4, angle_power=0.55) + column(6, angle_power=0.95)
    decision = evaluate_rank_decision(records, _rule())
    paths = write_rank_decision(decision, tmp_path)
    payload = json.loads(paths["rank_decision"].read_text(encoding="utf-8"))
    assert payload["verdict"] == "adopt_fixed_rank" and payload["chosen_rank"] == 6
    assert payload["rule"]["protected"][0] == {"trajectory_mode": "magnitude", "statistic": "delta"}
    assert "same generated dataset" in payload["pairing"]
    assert [c["rank"] for c in payload["columns"]] == [None, 4, 6]
    assert payload["columns"][2]["criteria"][0]["observations"]["threshold"] > 0
    frame = pd.read_csv(paths["rank_decision_csv"])
    assert list(frame["component_selection"]) == ["cv", "forced", "forced"]
    assert frame["chosen"].sum() == 1
    expected = {"gain_passed", "anchor_passed", "protected_passed", "target_rate", "anchor_angle_rate"}
    assert expected <= set(frame.columns)


# ── 3.4 ladder figure ─────────────────────────────────────────────────────────


def test_rank_ladder_is_written_for_a_rank_axis_and_not_otherwise(tmp_path: Path) -> None:
    from tests.test_study_design_point_report import _grid_records

    records = column(None, angle_power=0.5) + column(4, angle_power=0.6) + column(6, angle_power=0.9)
    frame = resolve_operating_by_design_point(records)
    out = render_rank_ladder(frame, tmp_path / "rank_ladder.png")
    assert out is not None and out.exists() and out.stat().st_size > 0

    rho_n = resolve_operating_by_design_point(_grid_records())
    assert render_rank_ladder(rho_n, tmp_path / "never.png") is None
    assert not (tmp_path / "never.png").exists()
    assert render_rank_ladder(pd.DataFrame(), tmp_path / "empty.png") is None


# ── 3.5 wiring ────────────────────────────────────────────────────────────────


def test_declared_forced_ranks_pass_the_production_guard_and_undeclared_ones_do_not() -> None:
    declared = column(None, angle_power=0.5) + column(6, angle_power=0.9)
    assert_production_component_selection(declared)
    frames = build_report_frames([], [], declared)
    assert set(frames.design_point_operating["component_selection"]) == {"cv", "forced"}

    undeclared = [record(rank=6, mode="orientation", effect=1.0, index=0, p={"angle": 0.01}, declare=False)]
    with pytest.raises(StudyReportError, match="undeclared forced"):
        assert_production_component_selection(undeclared)
    with pytest.raises(StudyReportError):
        build_report_frames([], [], undeclared)

    mismatched = [record(rank=6, mode="orientation", effect=1.0, index=0, p={"angle": 0.01}, selected_lv=5)]
    with pytest.raises(StudyReportError, match="undeclared forced"):
        assert_production_component_selection(mismatched)

    legacy = [record(rank=None, mode="orientation", effect=1.0, index=0, p={"angle": 0.01})]
    legacy[0].integration_metadata.pop("component_selection")
    assert_production_component_selection(legacy)


def test_design_point_power_degrades_gracefully_on_a_rank_only_grid(tmp_path: Path) -> None:
    records = column(None, angle_power=0.5) + column(4, angle_power=0.6) + column(6, angle_power=0.9)
    frame = resolve_operating_by_design_point(records)
    out = render_design_point_power(frame, tmp_path / "design_point_power.png")
    assert out.exists() and out.stat().st_size > 0


def test_baseline_readers_only_see_the_cv_column() -> None:
    from motco.simulations.study.summary import summarize_combined_rule, summarize_study

    records = column(None, angle_power=0.5) + column(4, angle_power=0.6) + column(6, angle_power=0.9)
    per_stat = summarize_study(records, alpha=0.05)
    combined = summarize_combined_rule(records, alpha=0.05)
    frames = build_report_frames(per_stat, combined, records)
    assert set(frames.power_curves["phase"]) == {"power_primary"}
    assert set(frames.specificity_matrix["phase"]) == {"power_primary"}
    assert frames.type_i_table.empty
