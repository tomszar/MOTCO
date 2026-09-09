from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from motco.simulations import SimulationReplicateResult, SimulationSummaryResult
from motco.simulations.study.report import (
    ReportFrames,
    StudyReportError,
    build_power_curves,
    build_report_frames,
    build_specificity_matrix,
    build_type_i_table,
    render_power_curves,
    render_specificity_matrix,
    render_type_i_plot,
    write_report_csvs,
)
from motco.simulations.study.summary import CombinedRuleSummary


def _record(
    cell_id: str,
    phase: str,
    mode: str | None,
    effect_size: float | None,
    varied_axis=None,
) -> SimulationReplicateResult:
    metadata = {}
    if mode is not None:
        metadata["trajectory_mode"] = mode
    if effect_size is not None:
        metadata["effect_size"] = effect_size
    metadata["varied_axis"] = varied_axis
    return SimulationReplicateResult(
        cell_id=cell_id,
        phase=phase,
        replicate_index=0,
        replicate_seed=0,
        generator_seed=0,
        evaluation_seed=0,
        parameter_signature="sig",
        status="completed",
        p_values={},
        cell_metadata=metadata,
    )


def _summary(cell_id: str, phase: str, statistic: str, rate: float, se: float) -> SimulationSummaryResult:
    return SimulationSummaryResult(
        cell_id=cell_id,
        phase=phase,
        statistic=statistic,
        alpha=0.05,
        completed_replicates=100,
        available_replicates=100,
        rejected_replicates=int(rate * 100),
        rejection_rate=rate,
        monte_carlo_se=se,
        unavailable_replicates=0,
    )


def _build_synthetic_summaries() -> tuple[list[SimulationSummaryResult], list[SimulationReplicateResult]]:
    records = [
        _record("null-none", "type_i_baseline", mode=None, effect_size=None),
        _record("null-translation", "type_i_baseline", mode="translation", effect_size=0.5),
        _record("magn-0.1", "power_primary", mode="magnitude", effect_size=0.1),
        _record("magn-0.5", "power_primary", mode="magnitude", effect_size=0.5),
        _record("orient-0.5", "power_primary", mode="orientation", effect_size=0.5),
        _record("ofat-magn", "power_ofat", mode="magnitude", effect_size=0.5, varied_axis="generator.n_samples"),
    ]
    summaries: list[SimulationSummaryResult] = []
    for stat in ("delta", "angle", "shape"):
        summaries.append(_summary("null-none", "type_i_baseline", stat, 0.05, 0.02))
        summaries.append(_summary("null-translation", "type_i_baseline", stat, 0.06, 0.02))
        summaries.append(_summary("magn-0.1", "power_primary", stat, 0.15, 0.04))
        summaries.append(_summary("magn-0.5", "power_primary", stat, 0.90, 0.03))
        summaries.append(_summary("orient-0.5", "power_primary", stat, 0.70, 0.05))
        summaries.append(_summary("ofat-magn", "power_ofat", stat, 0.85, 0.04))
    return summaries, records


def test_specificity_matrix_has_one_row_per_mode_statistic() -> None:
    summaries, records = _build_synthetic_summaries()
    frame = build_specificity_matrix(summaries, records)
    assert {"trajectory_mode", "statistic", "rejection_rate", "monte_carlo_se"}.issubset(frame.columns)
    pairs = set(zip(frame["trajectory_mode"], frame["statistic"]))
    assert pairs == {
        ("none", "delta"),
        ("none", "angle"),
        ("none", "shape"),
        ("translation", "delta"),
        ("translation", "angle"),
        ("translation", "shape"),
        ("magnitude", "delta"),
        ("magnitude", "angle"),
        ("magnitude", "shape"),
        ("orientation", "delta"),
        ("orientation", "angle"),
        ("orientation", "shape"),
    }
    # For magnitude, top effect size = 0.5 → rate = 0.90
    magnitude_delta = frame.query("trajectory_mode=='magnitude' and statistic=='delta'").iloc[0]
    assert magnitude_delta["rejection_rate"] == 0.90
    assert magnitude_delta["effect_size"] == 0.5


def test_power_curves_skip_ofat_cells() -> None:
    summaries, records = _build_synthetic_summaries()
    frame = build_power_curves(summaries, records)
    assert {"trajectory_mode", "statistic", "effect_size", "rejection_rate"}.issubset(frame.columns)
    # ofat cells are excluded
    assert (frame["phase"] != "power_ofat").all()
    # for magnitude delta, both effect sizes present
    magn_delta = frame.query("trajectory_mode=='magnitude' and statistic=='delta'")
    assert sorted(magn_delta["effect_size"].tolist()) == [0.1, 0.5]


def test_type_i_table_includes_per_statistic_and_combined_columns() -> None:
    summaries, records = _build_synthetic_summaries()
    combined = [
        CombinedRuleSummary(
            cell_id="null-none",
            phase="type_i_baseline",
            alpha=0.05,
            completed_replicates=100,
            available_replicates=100,
            rejected_replicates=12,
            rejection_rate=0.12,
            monte_carlo_se=0.03,
            statistics=("delta", "angle", "shape"),
        ),
        CombinedRuleSummary(
            cell_id="null-translation",
            phase="type_i_baseline",
            alpha=0.05,
            completed_replicates=100,
            available_replicates=100,
            rejected_replicates=14,
            rejection_rate=0.14,
            monte_carlo_se=0.03,
            statistics=("delta", "angle", "shape"),
        ),
    ]
    frame = build_type_i_table(summaries, combined, records)
    for column in (
        "delta_rate", "delta_se", "angle_rate", "angle_se", "shape_rate", "shape_se",
        "combined_rate", "combined_se", "trajectory_mode",
    ):
        assert column in frame.columns
    none_row = frame.query("cell_id=='null-none'").iloc[0]
    assert none_row["combined_rate"] == 0.12
    translation_row = frame.query("cell_id=='null-translation'").iloc[0]
    assert translation_row["trajectory_mode"] == "translation"


def test_write_report_csvs_produces_three_files(tmp_path: Path) -> None:
    summaries, records = _build_synthetic_summaries()
    combined: list[CombinedRuleSummary] = []
    frames = ReportFrames(
        specificity_matrix=build_specificity_matrix(summaries, records),
        power_curves=build_power_curves(summaries, records),
        type_i_table=build_type_i_table(summaries, combined, records),
    )
    paths = write_report_csvs(frames, tmp_path)
    for key in ("specificity_matrix", "power_curves", "type_i_table"):
        assert paths[key].exists()
        pd.read_csv(paths[key])  # parses cleanly


def test_figure_renderers_write_png(tmp_path: Path) -> None:
    summaries, records = _build_synthetic_summaries()
    combined: list[CombinedRuleSummary] = []
    specificity = build_specificity_matrix(summaries, records)
    power = build_power_curves(summaries, records)
    type_i = build_type_i_table(summaries, combined, records)

    a = render_specificity_matrix(specificity, tmp_path / "matrix.png")
    b = render_power_curves(power, tmp_path / "curves.png")
    c = render_type_i_plot(type_i, tmp_path / "type_i.png")
    for path in (a, b, c):
        assert path.exists()
        assert path.stat().st_size > 0


def test_report_frames_reject_records_whose_latent_rank_was_forced() -> None:
    """A rank-diagnostic record must never be reported as a production result."""

    summaries, records = _build_synthetic_summaries()
    forced = replace(
        records[2],
        integration_metadata={"component_selection": "forced", "forced_components": 9},
    )
    contaminated = [*records[:2], forced, *records[3:]]

    with pytest.raises(StudyReportError, match="forced latent rank"):
        build_report_frames(summaries, [], contaminated)


def test_report_frames_accept_cross_validated_and_pre_marker_records() -> None:
    summaries, records = _build_synthetic_summaries()
    marked = [
        replace(record, integration_metadata={"component_selection": "cv"}) for record in records[:3]
    ]
    # The remaining records predate the marker and carry no key at all.
    frames = build_report_frames(summaries, [], [*marked, *records[3:]])

    assert not frames.specificity_matrix.empty


# --- report contract: echo, driver table, figure gating -------------------------


def _contract_config(*, n_jobs: int = 1, contract=None):
    from motco.simulations import SemiSyntheticTrajectoryParams, SimulationEvaluationParams
    from motco.simulations.study import MatchedSeedPolicy, ReportContract, StudyConfig

    return StudyConfig(
        generator=SemiSyntheticTrajectoryParams(seed=2, trajectory_mode="magnitude", n_samples=60),
        evaluation=SimulationEvaluationParams(integration_method="pls", permutations=9, seed=3, n_jobs=n_jobs),
        trajectory_modes=("magnitude", "orientation", "shape", "translation"),
        effect_sizes=(0.0, 0.5),
        matched_seeds=MatchedSeedPolicy(enabled=True, primary_family="fam"),
        report_contract=(
            contract
            if contract is not None
            else ReportContract(driver_component="observed", n_jobs_override="forbid")
        ),
    )


def _contract_records(n_jobs_values=(1, 1, 1), *, anchor: bool = True) -> list[SimulationReplicateResult]:
    records = []
    for index, n_jobs in enumerate(n_jobs_values):
        meta = {"trajectory_mode": "none", "effect_size": 0.0, "varied_axis": None}
        if anchor:
            meta["zero_effect_anchor"] = True
            meta["resolves_modes"] = ["magnitude", "orientation", "shape", "translation"]
        records.append(
            replace(
                _record("anchor-cell", "power_primary", mode="none", effect_size=0.0),
                replicate_index=index,
                cell_metadata=meta,
                runtime_metadata={"runtime_seconds": 0.1, "n_jobs": n_jobs},
            )
        )
    records.append(
        replace(
            _record("null-none", "type_i_baseline", mode=None, effect_size=None),
            runtime_metadata={"runtime_seconds": 0.1, "n_jobs": n_jobs_values[0]},
        )
    )
    return records


def test_report_contract_echo_names_component_anchor_and_uniform_n_jobs(tmp_path: Path) -> None:
    import json

    from motco.simulations.study.report import build_report_contract_echo, write_report_contract

    config = _contract_config()
    records = _contract_records()
    echo = build_report_contract_echo(config, records)
    assert echo["driver_component"] == "observed"
    assert echo["cross_replicate_driver_agreement"] == "descriptive"
    assert echo["n_jobs_override"] == "forbid"
    assert echo["n_jobs"] == 1 and echo["config_n_jobs"] == 1
    assert echo["zero_effect_anchor"] == {
        "cell_id": "anchor-cell",
        "resolves_modes": ["magnitude", "orientation", "shape", "translation"],
        "counted_as": 1,
    }
    statements = echo["statements"]
    assert "not a driver-stability claim" in statements["cross_replicate_driver_agreement"]
    assert "anchor-cell" in statements["zero_effect_anchor"] and "one measurement" in statements["zero_effect_anchor"]
    assert "'observed'" in statements["driver_component"]

    path = write_report_contract(config, records, tmp_path)
    assert path == tmp_path / "report_contract.json"
    assert json.loads(path.read_text(encoding="utf-8")) == echo


def test_report_contract_echo_refuses_non_uniform_n_jobs() -> None:
    from motco.simulations.study.report import build_report_contract_echo

    with pytest.raises(ValueError, match=r"more than one n_jobs value: \[1, 4\]"):
        build_report_contract_echo(_contract_config(), _contract_records((1, 4, 1)))


def test_report_contract_echo_without_anchor_or_recorded_n_jobs() -> None:
    from motco.simulations.study.report import build_report_contract_echo

    records = [replace(r, runtime_metadata={"runtime_seconds": 0.1}) for r in _contract_records(anchor=False)]
    echo = build_report_contract_echo(_contract_config(), records)
    assert echo["n_jobs"] is None and echo["zero_effect_anchor"] is None
    assert "no n_jobs value" in echo["statements"]["n_jobs"]
    assert "No shared zero-effect anchor" in echo["statements"]["zero_effect_anchor"]


def test_report_contract_echo_requires_a_contract() -> None:
    from motco.simulations.study.report import build_report_contract_echo

    config = replace(_contract_config(), report_contract=None)
    with pytest.raises(StudyReportError, match="report_contract"):
        build_report_contract_echo(config, _contract_records())


def _attribution_frame() -> pd.DataFrame:
    rows = []
    for component, precision in (("observed", 0.5), ("pls_captured", 0.15), ("residual", 0.3)):
        rows.append(
            {
                "trajectory_mode": "orientation",
                "effect_size": 1.0,
                "cell_id": "cell",
                "transition_id": "0->1",
                "component": component,
                "eligible_replicates": 3,
                "computed_replicates": 3,
                "failed_replicates": 0,
                "availability_rate": 1.0,
                "retention_cosine_mean": 0.8,
                "retention_norm_ratio_mean": 0.7,
                "residual_fraction_mean": 0.3,
                "precision_mean": precision,
                "recall_mean": 0.1,
                "selected_count_mean": 3.0,
                "bootstrap_sign_stability_mean": 0.9,
                "bootstrap_top_k_frequency_mean": 0.6,
                "top_k_jaccard": 0.5,
                "sign_agreement": 0.8,
            }
        )
    return pd.DataFrame(rows)


def test_write_driver_report_restricts_to_the_declared_component(tmp_path: Path) -> None:
    from motco.simulations.study import ReportContract
    from motco.simulations.study.report import write_driver_report

    path = write_driver_report(_attribution_frame(), ReportContract(driver_component="residual"), tmp_path)
    frame = pd.read_csv(path)
    assert path.name == "driver_report.csv"
    assert len(frame) == 1 and frame.iloc[0]["precision_mean"] == pytest.approx(0.3)
    assert "top_k_jaccard" not in frame.columns and "component" not in frame.columns


def test_attribution_figure_under_a_contract_and_without(tmp_path: Path) -> None:
    from motco.simulations.study import ReportContract
    from motco.simulations.study.report import render_attribution_stability

    frame = _attribution_frame()
    legacy = render_attribution_stability(frame, tmp_path / "legacy.png")
    contract = render_attribution_stability(
        frame, tmp_path / "contract.png", contract=ReportContract(driver_component="pls_captured")
    )
    assert legacy.exists() and contract.exists()
    # Different series and title → different rendering.
    assert legacy.read_bytes() != contract.read_bytes()
    # The no-contract path is the historical figure: identical bytes on a re-render.
    again = render_attribution_stability(frame, tmp_path / "legacy2.png")
    assert again.read_bytes() == legacy.read_bytes()
    empty = render_attribution_stability(
        pd.DataFrame(), tmp_path / "empty.png", contract=ReportContract(driver_component="observed")
    )
    assert empty.exists()


def test_attribution_figure_titles_follow_the_contract(monkeypatch, tmp_path: Path) -> None:
    """The contract path plots only within-replicate series under the declared title."""

    import matplotlib.pyplot as plt

    from motco.simulations.study import ReportContract
    from motco.simulations.study.report import render_attribution_stability

    seen: dict[str, object] = {}
    original_subplots = plt.subplots

    def capture(*args, **kwargs):
        fig, ax = original_subplots(*args, **kwargs)
        seen["ax"] = ax
        return fig, ax

    monkeypatch.setattr(plt, "subplots", capture)
    render_attribution_stability(
        _attribution_frame(), tmp_path / "c.png", contract=ReportContract(driver_component="observed")
    )
    ax = seen["ax"]
    assert ax.get_title() == "Within-replicate bootstrap stability (observed component)"  # type: ignore[attr-defined]
    labels = [line.get_label() for line in ax.get_lines()]  # type: ignore[attr-defined]
    assert labels == ["bootstrap sign stability", "bootstrap top-k selection frequency"]

    render_attribution_stability(_attribution_frame(), tmp_path / "l.png")
    ax = seen["ax"]
    assert ax.get_title() == "Attribution stability (observed component)"  # type: ignore[attr-defined]
    labels = [line.get_label() for line in ax.get_lines()]  # type: ignore[attr-defined]
    assert labels == ["cross-replicate top-k Jaccard", "cross-replicate sign agreement", "bootstrap sign stability"]
