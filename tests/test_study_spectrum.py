"""Per-cell eigengap summaries and eigengap-stratified power.

Records are built by hand with known eigengaps and known rejection outcomes, so
each stratum's expected rate is constructed rather than inferred.
"""

from __future__ import annotations

import pandas as pd
import pytest

from motco.simulations.grid import SimulationReplicateResult
from motco.simulations.study.report import build_report_frames, write_report_csvs
from motco.simulations.study.spectrum import (
    group_eigengaps,
    has_spectrum,
    pooled_eigengap,
    stratify_power_by_eigengap,
    summarize_config_spectrum,
)


def spectrum_block(pooled: float | None, groups: dict[str, float] | None = None) -> dict:
    def entry(gap: float | None) -> dict:
        return {
            "n_points": 4,
            "n_dimensions": 5,
            "total_variance": 0.0 if gap is None else 4.0,
            "spectrum": [] if gap is None else [0.5, 0.5 - gap, gap],
            "relative_eigengap": gap,
        }

    return {
        "version": 1,
        "pooled": entry(pooled),
        "groups": {name: entry(gap) for name, gap in (groups or {"A": 0.1, "B": 0.2}).items()},
    }


def record(
    *,
    cell_id: str = "orientation-cell",
    mode: str = "orientation",
    phase: str = "power_primary",
    replicate_index: int = 0,
    eigengap: float | None = 0.05,
    p_value: float | None = 0.01,
    with_spectrum: bool = True,
    effect_size: float = 1.0,
) -> SimulationReplicateResult:
    return SimulationReplicateResult(
        cell_id=cell_id,
        phase=phase,
        replicate_index=replicate_index,
        replicate_seed=replicate_index,
        generator_seed=replicate_index,
        evaluation_seed=replicate_index,
        parameter_signature="sig",
        status="completed",
        p_values={} if p_value is None else {"angle": p_value},
        pair_statistics={"angle": 40.0},
        cell_metadata={"trajectory_mode": mode, "effect_size": effect_size},
        runtime_metadata={"permutations": 199},
        config_spectrum=spectrum_block(eigengap) if with_spectrum else {},
    )


# ── Record accessors ──────────────────────────────────────────────────────────


def test_accessors_separate_a_missing_field_from_a_degenerate_spectrum() -> None:
    missing = record(with_spectrum=False)
    degenerate = record(eigengap=None)

    assert pooled_eigengap(missing) is None and not has_spectrum(missing)
    assert pooled_eigengap(degenerate) is None and has_spectrum(degenerate)
    assert group_eigengaps(record(eigengap=0.3)) == {"A": 0.1, "B": 0.2}


# ── 4.1 Per-cell summaries ────────────────────────────────────────────────────


def test_cell_summary_reports_pooled_and_per_group_eigengaps() -> None:
    records = [
        record(replicate_index=index, eigengap=gap)
        for index, gap in enumerate((0.02, 0.04, 0.06, 0.08))
    ]

    frame = summarize_config_spectrum(records)

    assert set(frame["configuration"]) == {"pooled", "A", "B"}
    pooled = frame[frame["configuration"] == "pooled"].iloc[0]
    assert pooled["n_available"] == 4
    assert pooled["mean_eigengap"] == pytest.approx(0.05)
    assert pooled["median_eigengap"] == pytest.approx(0.05)
    assert pooled["min_eigengap"] == pytest.approx(0.02)
    assert pooled["max_eigengap"] == pytest.approx(0.08)
    assert frame[frame["configuration"] == "A"].iloc[0]["mean_eigengap"] == pytest.approx(0.1)


def test_cell_summary_counts_degenerate_and_missing_spectra_separately() -> None:
    records = [
        record(replicate_index=0, eigengap=0.05),
        record(replicate_index=1, eigengap=None),
        record(replicate_index=2, with_spectrum=False),
    ]

    pooled = summarize_config_spectrum(records)
    pooled = pooled[pooled["configuration"] == "pooled"].iloc[0]

    assert pooled["n_replicates"] == 3
    assert pooled["n_recorded"] == 2  # the third predates the field
    assert pooled["n_available"] == 1  # the second is degenerate
    assert pooled["mean_eigengap"] == pytest.approx(0.05)


# ── 4.2 Stratified power ──────────────────────────────────────────────────────


def test_orientation_power_is_stratified_by_recorded_eigengap() -> None:
    """Constructed so the strata carry known rates: 0.0, 0.5, 1.0."""

    gaps = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
    # The bottom tercile never rejects, the middle rejects once, the top always
    # rejects — the monotone pattern the audit predicts, built by hand.
    p_values = [0.9, 0.9, 0.9, 0.9, 0.9, 0.01, 0.01, 0.01, 0.01]
    records = [
        record(replicate_index=index, eigengap=gap, p_value=p)
        for index, (gap, p) in enumerate(zip(gaps, p_values))
    ]

    frame = stratify_power_by_eigengap(records, statistics=("angle",))

    assert list(frame["stratum"]) == [0, 1, 2]
    assert list(frame["n_replicates"]) == [3, 3, 3]
    assert list(frame["rejection_rate"]) == pytest.approx([0.0, 1 / 3, 1.0])
    assert frame.iloc[0]["eigengap_high"] == pytest.approx(0.03)
    assert frame.iloc[2]["eigengap_low"] == pytest.approx(0.07)
    assert frame.iloc[2]["monte_carlo_se"] == pytest.approx(0.0)
    assert frame.iloc[1]["monte_carlo_se"] == pytest.approx((1 / 3 * 2 / 3 / 3) ** 0.5)
    assert set(frame["status"]) == {"ok"}


def test_stratification_reads_only_the_power_cells_it_is_declared_for() -> None:
    records = [
        record(replicate_index=0, eigengap=0.05),
        record(cell_id="magnitude-cell", mode="magnitude", replicate_index=1, eigengap=0.5),
        record(cell_id="type-i", mode="none", phase="type_i_baseline", replicate_index=2, eigengap=0.4),
    ]

    frame = stratify_power_by_eigengap(records, statistics=("angle",))

    assert set(frame["cell_id"]) == {"orientation-cell"}


def test_cells_without_recorded_spectra_are_reported_as_unavailable() -> None:
    records = [
        record(replicate_index=index, with_spectrum=False, p_value=0.01) for index in range(4)
    ]

    frame = stratify_power_by_eigengap(records, statistics=("angle",))

    assert list(frame["status"]) == ["unavailable"]
    assert frame.iloc[0]["rejection_rate"] is None
    assert frame.iloc[0]["n_replicates"] == 4


def test_legacy_records_leave_every_pre_existing_table_unchanged(tmp_path) -> None:
    """Reporting a pre-spectrum record set still produces the old tables verbatim."""

    from motco.simulations.study.report import (
        build_power_curves,
        build_specificity_matrix,
        build_type_i_table,
    )

    records = [record(replicate_index=index, with_spectrum=False) for index in range(4)]
    frames = build_report_frames([], [], records)

    pd.testing.assert_frame_equal(frames.specificity_matrix, build_specificity_matrix([], records))
    pd.testing.assert_frame_equal(frames.power_curves, build_power_curves([], records))
    pd.testing.assert_frame_equal(frames.type_i_table, build_type_i_table([], [], records))

    paths = write_report_csvs(frames, tmp_path / "report")
    assert set(frames.eigengap_stratified_power["status"]) == {"unavailable"}
    assert paths["config_spectrum"].exists()
    assert paths["eigengap_stratified_power"].exists()
