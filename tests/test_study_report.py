from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from motco.simulations import SimulationReplicateResult, SimulationSummaryResult
from motco.simulations.study.report import (
    ReportFrames,
    build_power_curves,
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
        intersim_seed=0,
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
        _record("ofat-magn", "power_ofat", mode="magnitude", effect_size=0.5, varied_axis="intersim.n_sample"),
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
