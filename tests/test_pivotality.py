"""Pivotality analysis over synthetic replicate records.

Every test here builds records by hand rather than running a study, so the
expected association is constructed and known.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from motco.simulations.grid import SimulationReplicateResult
from motco.simulations.pivotality import (
    PivotalityError,
    association_table,
    cell_key,
    rejection_split_table,
    replicate_z,
    spectrum_association_table,
    standardized_counterfactual_table,
    write_pivotality_tables,
)


def make_record(
    *,
    cell_id: str,
    trajectory_mode: str,
    observed: float,
    null_mean: float,
    null_sd: float,
    null_q95: float,
    p_value: float,
    replicate_index: int = 0,
    phase: str = "power_primary",
    effect_size: float = 1.0,
    statistic: str = "angle",
    eigengap: float | None = None,
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
        p_values={statistic: p_value},
        pair_statistics={statistic: observed},
        cell_metadata={"trajectory_mode": trajectory_mode, "effect_size": effect_size},
        runtime_metadata={"permutations": 199},
        null_summary={
            statistic: {
                "count": 199.0,
                "mean": null_mean,
                "sd": null_sd,
                "q50": null_mean,
                "q90": null_q95 * 0.9,
                "q95": null_q95,
                "q99": null_q95 * 1.1,
            }
        },
        config_spectrum={} if eigengap is None else {
            "version": 1,
            "pooled": {
                "n_points": 4,
                "n_dimensions": 5,
                "total_variance": 3.0,
                "spectrum": [0.5, 0.5 - eigengap, eigengap],
                "relative_eigengap": eigengap,
            },
            "groups": {},
        },
    )


#: Signal carried above each tracking replicate's own null mean, in null sd units.
TRACKING_SIGNAL = 8.0
TRACKING_SD = 4.0


def tracking_cell(cell_id: str = "tracking", mode: str = "orientation", n: int = 40):
    """The non-pivotal case: the whole null slides with the observed statistic.

    The raw observed angle wanders over 30–69° across replicates, but every
    replicate carries the *same* signal above its own null mean. Its critical
    value slides with it and sits above the observed value, so the test never
    rejects however large the raw angle gets — the power-costing shape the
    diagnostic is looking for.
    """

    records = []
    for index in range(n):
        observed = 30.0 + index
        null_mean = observed - TRACKING_SIGNAL
        records.append(
            make_record(
                cell_id=cell_id,
                trajectory_mode=mode,
                observed=observed,
                null_mean=null_mean,
                null_sd=TRACKING_SD,
                # The per-replicate null is heavy-tailed, so its own q95 sits
                # above the observed value even though the signal is a steady
                # 2 sd. This is what a control-calibrated z can see and the
                # replicate's own critical value cannot.
                null_q95=null_mean + 12.0,
                p_value=0.40,
                replicate_index=index,
            )
        )
    return records


def pivotal_cell(cell_id: str = "pivotal", mode: str = "magnitude", n: int = 40, seed: int = 7):
    """The pivotal case: the null is a property of the design, not the replicate."""

    rng = np.random.default_rng(seed)
    records = []
    for index in range(n):
        observed = float(30.0 + 12.0 * rng.standard_normal())
        records.append(
            make_record(
                cell_id=cell_id,
                trajectory_mode=mode,
                observed=observed,
                null_mean=float(20.0 + 0.01 * rng.standard_normal()),
                null_sd=float(6.0 + 0.01 * rng.standard_normal()),
                null_q95=float(31.0 + 0.01 * rng.standard_normal()),
                p_value=0.01 if observed > 31.0 else 0.40,
                replicate_index=index,
            )
        )
    return records


def rows_for(rows, *, mode: str, statistic: str = "angle", null_target: str | None = None):
    matched = [
        row
        for row in rows
        if row.cell.trajectory_mode == mode
        and row.statistic == statistic
        and (null_target is None or row.null_target == null_target)
    ]
    assert len(matched) == 1, f"expected one row, got {len(matched)}"
    return matched[0]


# --- association ------------------------------------------------------------


def test_association_detects_a_null_that_tracks_the_observed_statistic() -> None:
    rows = association_table(tracking_cell(), statistics=("angle",))

    for target in ("mean", "q95"):
        row = rows_for(rows, mode="orientation", null_target=target)
        assert row.n_replicates == 40
        assert row.correlation == pytest.approx(1.0)
        assert row.slope == pytest.approx(1.0)
    # The spread is constant by construction: no association, and undefined
    # correlation rather than a fabricated zero.
    sd_row = rows_for(rows, mode="orientation", null_target="sd")
    assert sd_row.correlation is None
    # A constant regressed on a varying x has slope 0 — the null's spread does
    # not move with the signal, which is reported rather than left undefined.
    assert sd_row.slope == pytest.approx(0.0)


def test_association_is_absent_for_a_pivotal_statistic() -> None:
    rows = association_table(pivotal_cell(), statistics=("angle",))

    for target in ("mean", "sd", "q95"):
        row = rows_for(rows, mode="magnitude", null_target=target)
        assert row.correlation is not None
        assert abs(row.correlation) < 0.5
        # The interval must be reported so a weak association is not over-read.
        assert row.correlation_ci_low is not None and row.correlation_ci_high is not None
        assert row.correlation_ci_low <= row.correlation <= row.correlation_ci_high


def test_association_reports_every_cell_and_statistic_separately() -> None:
    records = tracking_cell() + pivotal_cell()
    rows = association_table(records, statistics=("angle",))

    modes = {row.cell.trajectory_mode for row in rows}
    assert modes == {"orientation", "magnitude"}
    assert len(rows) == 2 * len(("mean", "sd", "q95"))
    assert rows_for(rows, mode="orientation", null_target="q95").correlation == pytest.approx(1.0)
    assert abs(rows_for(rows, mode="magnitude", null_target="q95").correlation) < 0.5


def test_association_skips_records_missing_a_null_summary() -> None:
    records = tracking_cell(n=5)
    stripped = [
        SimulationReplicateResult(**{**record.__dict__, "null_summary": {}}) for record in records[:2]
    ]
    rows = association_table(stripped + records[2:], statistics=("angle",))

    assert rows_for(rows, mode="orientation", null_target="mean").n_replicates == 3


# --- rejection split --------------------------------------------------------


def test_rejection_split_reproduces_a_constructed_inversion() -> None:
    """Non-rejecting replicates carry the larger observed angle — the Phase 4 pattern."""

    records = [
        # Rejects: small angle, but a far smaller bar.
        make_record(
            cell_id="c", trajectory_mode="orientation", observed=20.0, null_mean=8.0,
            null_sd=3.0, null_q95=14.0, p_value=0.01, replicate_index=0,
        ),
        make_record(
            cell_id="c", trajectory_mode="orientation", observed=22.0, null_mean=9.0,
            null_sd=3.0, null_q95=15.0, p_value=0.02, replicate_index=1,
        ),
        # Does not reject: larger angle, and a bar that moved further still.
        make_record(
            cell_id="c", trajectory_mode="orientation", observed=60.0, null_mean=55.0,
            null_sd=9.0, null_q95=70.0, p_value=0.40, replicate_index=2,
        ),
        make_record(
            cell_id="c", trajectory_mode="orientation", observed=64.0, null_mean=58.0,
            null_sd=9.0, null_q95=74.0, p_value=0.55, replicate_index=3,
        ),
    ]
    row = rows_for(rejection_split_table(records, statistics=("angle",)), mode="orientation")

    assert row.n_rejecting == 2
    assert row.n_non_rejecting == 2
    assert row.mean_observed_rejecting == pytest.approx(21.0)
    assert row.mean_observed_non_rejecting == pytest.approx(62.0)
    assert row.inverted is True
    # The follow-up the spec requires: the larger observed statistic came with a
    # proportionally larger critical value.
    assert row.mean_critical_rejecting == pytest.approx(14.5)
    assert row.mean_critical_non_rejecting == pytest.approx(72.0)


def test_rejection_split_reports_no_inversion_when_rejections_track_the_signal() -> None:
    row = rows_for(rejection_split_table(pivotal_cell(), statistics=("angle",)), mode="magnitude")

    assert row.inverted is False
    assert row.mean_observed_rejecting > row.mean_observed_non_rejecting


def test_rejection_split_rejects_an_invalid_alpha() -> None:
    with pytest.raises(PivotalityError, match="alpha"):
        rejection_split_table(pivotal_cell(), alpha=1.5)


# --- standardized counterfactual -------------------------------------------


def null_control_cell(cell_id: str = "control", mode: str = "none", n: int = 60, seed: int = 3):
    """A null control: observed and null drawn from the same distribution."""

    rng = np.random.default_rng(seed)
    records = []
    for index in range(n):
        observed = float(20.0 + 6.0 * rng.standard_normal())
        records.append(
            make_record(
                cell_id=cell_id,
                trajectory_mode=mode,
                observed=observed,
                null_mean=20.0,
                null_sd=6.0,
                null_q95=30.0,
                p_value=0.50,
                replicate_index=index,
                phase="type_i_baseline",
                effect_size=0.0,
            )
        )
    return records


def test_standardized_counterfactual_recovers_rejections_only_where_the_null_tracks() -> None:
    """The tracking cell's signal is invisible as-specified but separates in ``z``.

    Its observed angles run far above the controls' while its p-values never
    reject, because each replicate's bar moved with it. Standardizing against a
    reference borrowed from the controls exposes that separation.
    """

    records = null_control_cell() + tracking_cell()
    rows = standardized_counterfactual_table(records, statistics=("angle",))

    tracking = rows_for(rows, mode="orientation")
    control = rows_for(rows, mode="none")

    assert tracking.as_specified_rate == 0.0  # every p-value is 0.40
    assert tracking.standardized_rate == 1.0
    assert tracking.mean_z == pytest.approx(TRACKING_SIGNAL / TRACKING_SD)
    assert not tracking.is_null_control

    # The control's own rate is reported, so a gain bought with an inflated
    # Type I rate would be visible. Calibrated at 1 - alpha on its own draws.
    assert control.is_null_control
    assert control.standardized_rate == pytest.approx(0.05, abs=0.02)
    assert control.z_threshold == tracking.z_threshold


def test_standardized_counterfactual_recovers_nothing_without_a_tracking_null() -> None:
    """A cell whose observed statistic sits inside the control ``z`` range gains nothing."""

    flat = [
        make_record(
            cell_id="flat", trajectory_mode="orientation", observed=20.0, null_mean=20.0,
            null_sd=6.0, null_q95=30.0, p_value=0.50, replicate_index=index,
        )
        for index in range(30)
    ]
    rows = standardized_counterfactual_table(null_control_cell() + flat, statistics=("angle",))

    assert rows_for(rows, mode="orientation").standardized_rate == 0.0


def test_standardized_counterfactual_reports_no_threshold_without_controls() -> None:
    rows = standardized_counterfactual_table(tracking_cell(), statistics=("angle",))
    row = rows_for(rows, mode="orientation")

    assert row.z_threshold is None
    assert row.standardized_rate is None
    assert row.as_specified_rate == 0.0


def test_replicate_z_is_none_without_a_usable_null_spread() -> None:
    record = make_record(
        cell_id="c", trajectory_mode="orientation", observed=20.0, null_mean=10.0,
        null_sd=0.0, null_q95=15.0, p_value=0.10,
    )
    assert replicate_z(record, "angle") is None
    assert replicate_z(record, "delta") is None


# --- the no-op that the counterfactual must not be mistaken for ---------------


def test_within_replicate_studentization_leaves_the_p_value_unchanged() -> None:
    """Standardizing both sides by the same constants is a monotone no-op.

    The cross-replicate counterfactual exists precisely because this does not
    work; the test pins it so the analysis cannot be misread as recommending
    within-replicate studentization.
    """

    from motco.simulations.evaluation import _empirical_p_value

    rng = np.random.default_rng(11)
    null = (40.0 + 9.0 * rng.standard_normal(999)).tolist()
    observed = 57.0

    baseline = _empirical_p_value(null, observed)

    null_mean = float(np.mean(null))
    null_sd = float(np.std(null, ddof=1))
    studentized_null = [(value - null_mean) / null_sd for value in null]
    studentized_observed = (observed - null_mean) / null_sd

    assert _empirical_p_value(studentized_null, studentized_observed) == baseline
    # Any positive affine rescaling behaves the same way.
    assert _empirical_p_value([3.0 * v + 7.0 for v in null], 3.0 * observed + 7.0) == baseline
    assert 0.0 < baseline <= 1.0


# --- plumbing ---------------------------------------------------------------


def test_cell_key_falls_back_to_none_for_the_type_i_baseline() -> None:
    record = SimulationReplicateResult(
        cell_id="type_i_baseline-x",
        phase="type_i_baseline",
        replicate_index=0,
        replicate_seed=0,
        generator_seed=0,
        evaluation_seed=0,
        parameter_signature="sig",
        status="completed",
        cell_metadata={"varied_axis": None, "varied_value": None},
    )
    key = cell_key(record)

    assert key.trajectory_mode == "none"
    assert key.effect_size is None
    assert key.label == "none@na[type_i_baseline]"


def test_failed_records_are_excluded_from_every_table() -> None:
    failed = SimulationReplicateResult(
        cell_id="tracking",
        phase="power_primary",
        replicate_index=99,
        replicate_seed=99,
        generator_seed=99,
        evaluation_seed=99,
        parameter_signature="sig",
        status="failed",
        cell_metadata={"trajectory_mode": "orientation", "effect_size": 1.0},
    )
    rows = association_table([*tracking_cell(n=5), failed], statistics=("angle",))

    assert rows_for(rows, mode="orientation", null_target="mean").n_replicates == 5


def test_write_pivotality_tables_emits_every_csv(tmp_path) -> None:
    import csv

    records = null_control_cell(n=10) + tracking_cell(n=10)
    paths = write_pivotality_tables(records, tmp_path, statistics=("angle",))

    assert set(paths) == {
        "pivotality_association",
        "pivotality_rejection_split",
        "pivotality_spectrum",
        "pivotality_standardized",
    }
    for path in paths.values():
        with open(path, encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert rows
        assert {"cell_id", "phase", "trajectory_mode", "effect_size", "statistic"} <= set(rows[0])
    with open(paths["pivotality_standardized"], encoding="utf-8", newline="") as handle:
        standardized = list(csv.DictReader(handle))
    assert any(row["trajectory_mode"] == "none" for row in standardized)
    assert all(math.isfinite(float(row["z_threshold"])) for row in standardized)


# --- eigengap covariate ------------------------------------------------------


def eigengap_cell(cell_id: str = "orientation", mode: str = "orientation", n: int = 40):
    """The audit's shape: a narrow configuration carries a wide null.

    The eigengap contracts as the index rises and the null width grows with it,
    so the association is negative and near-perfect on ranks by construction.
    """

    records = []
    for index in range(n):
        eigengap = 0.12 - 0.002 * index
        null_q95 = 10.0 * (0.12 / eigengap) ** 2
        records.append(
            make_record(
                cell_id=cell_id,
                trajectory_mode=mode,
                observed=45.0,
                null_mean=null_q95 * 0.6,
                null_sd=null_q95 * 0.2,
                null_q95=null_q95,
                p_value=0.01 if null_q95 < 45.0 else 0.40,
                replicate_index=index,
                eigengap=eigengap,
            )
        )
    return records


def test_eigengap_association_is_negative_when_narrow_geometry_widens_the_null() -> None:
    rows = spectrum_association_table(eigengap_cell(), statistics=("angle",), null_targets=("q95",))
    row = rows_for(rows, mode="orientation", null_target="q95")

    assert row.status == "ok"
    assert row.n_replicates == 40
    assert row.spearman == pytest.approx(-1.0)
    assert row.log_log_pearson is not None and row.log_log_pearson < -0.99
    # The construction sets q95 proportional to eigengap^-2.
    assert row.log_log_slope == pytest.approx(-2.0, rel=1e-6)
    assert row.mean_eigengap == pytest.approx(
        float(np.mean([0.12 - 0.002 * index for index in range(40)]))
    )


def test_eigengap_association_is_near_zero_when_the_null_does_not_track_geometry() -> None:
    """A null width drawn independently of the geometry shows no association."""

    rng = np.random.default_rng(4)
    records = [
        make_record(
            cell_id="flat",
            trajectory_mode="magnitude",
            observed=45.0,
            null_mean=20.0,
            null_sd=6.0,
            null_q95=float(30.0 + 5.0 * rng.standard_normal()),
            p_value=0.01,
            replicate_index=index,
            eigengap=0.05 + 0.001 * index,
        )
        for index in range(60)
    ]

    row = rows_for(
        spectrum_association_table(records, statistics=("angle",), null_targets=("q95",)),
        mode="magnitude",
        null_target="q95",
    )

    assert row.status == "ok"
    assert abs(row.spearman) < 0.25
    assert row.spearman_ci_low < 0.0 < row.spearman_ci_high


def test_eigengap_association_is_unavailable_without_recorded_spectra() -> None:
    row = rows_for(
        spectrum_association_table(tracking_cell(), statistics=("angle",), null_targets=("q95",)),
        mode="orientation",
        null_target="q95",
    )

    assert row.status == "unavailable"
    assert row.n_replicates == 0
    assert row.n_records_with_spectrum == 0
    assert row.spearman is None and row.log_log_pearson is None


def test_degenerate_eigengaps_are_dropped_from_the_log_log_fit_only() -> None:
    records = eigengap_cell(n=10)
    records.append(
        make_record(
            cell_id="orientation",
            trajectory_mode="orientation",
            observed=45.0,
            null_mean=30.0,
            null_sd=9.0,
            null_q95=60.0,
            p_value=0.40,
            replicate_index=99,
            eigengap=0.0,
        )
    )

    row = rows_for(
        spectrum_association_table(records, statistics=("angle",), null_targets=("q95",)),
        mode="orientation",
        null_target="q95",
    )

    # The rank correlation keeps all 11 replicates; only the logarithm cannot
    # take the degenerate one.
    assert row.n_replicates == 11
    assert row.spearman is not None
    assert row.log_log_pearson is not None
