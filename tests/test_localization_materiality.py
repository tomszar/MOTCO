"""Materiality rules for off-diagonal localization.

The legacy ``absolute`` rule applies one cut to three statistics with
incommensurable native scales, which makes a small-scale statistic
structurally unclassifiable. The ``null_dispersion`` rule judges each in units
of its own zero-effect null.
"""

from __future__ import annotations

import pytest

from motco.simulations.grid import SimulationReplicateResult
from motco.simulations.study.phase4 import (
    DEFAULT_MATERIALITY_DISPERSION_UNITS,
    DEFAULT_MATERIALITY_THRESHOLD,
    Phase4SummaryError,
    localize_off_diagonal,
)

_LEGACY_COLUMNS = [
    "trajectory_mode",
    "effect_size",
    "statistic",
    "first_material_checkpoint",
    "classification",
    "materiality_threshold",
    "normalized_value",
    "normalized_null",
    "normalized_excess",
    "measurement_space",
]


def _scope(delta: float, angle: float | None, shape: float | None) -> dict:
    return {
        "path_lengths": {"A": 1.0, "B": 1.0},
        "delta": delta,
        "angle": angle,
        "shape": shape,
        "availability": {
            "delta": True,
            "angle": angle is not None,
            "shape": shape is not None,
        },
    }


def _records(
    *,
    mode_shape: float,
    anchor_shape: float,
    anchor_shape_spread: float,
    n: int = 8,
) -> list[SimulationReplicateResult]:
    """Records whose ``shape`` values give a chosen mean and dispersion.

    The anchor alternates around ``anchor_shape`` by ``anchor_shape_spread`` so
    its recorded ``sd`` is controllable; the mode's value is constant.
    """

    out: list[SimulationReplicateResult] = []

    def emit(cell_id: str, meta: dict, shape_values: list[float], phase: str) -> None:
        for index, shape in enumerate(shape_values):
            out.append(
                SimulationReplicateResult(
                    cell_id=cell_id,
                    phase=phase,
                    replicate_index=index,
                    replicate_seed=index,
                    generator_seed=index,
                    evaluation_seed=index,
                    parameter_signature="sig",
                    status="completed",
                    p_values={"delta": 0.01, "angle": 0.01, "shape": 0.01},
                    pair_statistics={"delta": 1.0, "angle": 1.0, "shape": shape},
                    realized_geometry={
                        "schema_version": 1,
                        "requested": {},
                        "checkpoints": {
                            "observed_standardized": {"joint": _scope(1.0, 1.0, shape)},
                        },
                    },
                    truth_metadata={},
                    runtime_metadata={"runtime_seconds": 0.1},
                    cell_metadata=meta,
                    integration_metadata={"integration_method": "pls"},
                    attribution_status="not_requested",
                    attribution_diagnostics={},
                )
            )

    emit(
        "mag",
        {"trajectory_mode": "magnitude", "effect_size": 1.0},
        [mode_shape] * n,
        "power_primary",
    )
    alternating = [
        anchor_shape + (anchor_shape_spread if i % 2 else -anchor_shape_spread)
        for i in range(n)
    ]
    emit(
        "anchor",
        {"trajectory_mode": "none", "effect_size": 0.0, "zero_effect_anchor": True},
        alternating,
        "power_primary",
    )
    return out


def _shape_row(frame):
    return frame[frame["statistic"] == "shape"].iloc[0]


def test_absolute_rule_misses_a_small_scale_response():
    """The defect: a real response an order of magnitude below the 0.05 cut."""

    records = _records(mode_shape=0.023, anchor_shape=0.006, anchor_shape_spread=0.001)
    row = _shape_row(localize_off_diagonal(records, rule="absolute"))
    assert row["classification"] == "not_material"
    assert row["first_material_checkpoint"] is None


def test_null_dispersion_rule_classifies_the_same_response():
    records = _records(mode_shape=0.023, anchor_shape=0.006, anchor_shape_spread=0.001)
    row = _shape_row(localize_off_diagonal(records, rule="null_dispersion"))
    assert row["classification"] == "sampling_or_preprocessing_associated"
    assert row["first_material_checkpoint"] == "observed_standardized"
    assert row["materiality_basis"] == "null_dispersion"
    # excess 0.017 over a dispersion of ~0.001 is many null widths
    assert row["excess_in_dispersion_units"] > DEFAULT_MATERIALITY_DISPERSION_UNITS
    assert row["null_dispersion"] == pytest.approx(0.001, rel=0.2)


def test_null_dispersion_rule_still_rejects_a_response_inside_the_null():
    """Commensurable does not mean permissive: a response within the null is not material."""

    records = _records(mode_shape=0.0065, anchor_shape=0.006, anchor_shape_spread=0.001)
    row = _shape_row(localize_off_diagonal(records, rule="null_dispersion"))
    assert row["classification"] == "not_material"


def test_degenerate_null_falls_back_to_the_dust_floor_without_dividing():
    """A zero-variance null is exceeded by any real difference (design D2a)."""

    records = _records(mode_shape=0.023, anchor_shape=0.0, anchor_shape_spread=0.0)
    row = _shape_row(localize_off_diagonal(records, rule="null_dispersion"))
    assert row["classification"] == "sampling_or_preprocessing_associated"
    assert row["materiality_basis"] == "degenerate_null_dust_floor"
    # no division was attempted against the degenerate dispersion
    assert row["excess_in_dispersion_units"] is None
    assert row["normalized_excess"] == pytest.approx(0.023)


def test_degenerate_null_does_not_promote_floating_point_dust():
    """Dust in, not-material out — the fallback is not a rubber stamp."""

    records = _records(mode_shape=1e-15, anchor_shape=0.0, anchor_shape_spread=0.0)
    row = _shape_row(localize_off_diagonal(records, rule="null_dispersion"))
    assert row["classification"] == "not_material"


def test_legacy_rule_is_the_default_and_keeps_the_legacy_schema():
    """Frozen reports must keep regenerating identically (design D3a)."""

    records = _records(mode_shape=0.023, anchor_shape=0.006, anchor_shape_spread=0.001)
    default = localize_off_diagonal(records)
    assert list(default.columns) == _LEGACY_COLUMNS
    assert default["materiality_threshold"].iloc[0] == DEFAULT_MATERIALITY_THRESHOLD

    explicit = localize_off_diagonal(records, rule="absolute")
    assert list(explicit.columns) == _LEGACY_COLUMNS
    assert default.equals(explicit)


def test_new_rule_records_its_reasoning_in_extra_columns():
    records = _records(mode_shape=0.023, anchor_shape=0.006, anchor_shape_spread=0.001)
    frame = localize_off_diagonal(records, rule="null_dispersion")
    for column in (
        "materiality_rule",
        "materiality_basis",
        "null_dispersion",
        "excess_in_dispersion_units",
    ):
        assert column in frame.columns
    assert set(frame["materiality_rule"]) == {"null_dispersion"}
    assert frame["materiality_threshold"].iloc[0] == DEFAULT_MATERIALITY_DISPERSION_UNITS


def test_explicit_threshold_overrides_the_per_rule_default():
    records = _records(mode_shape=0.023, anchor_shape=0.006, anchor_shape_spread=0.001)
    row = _shape_row(
        localize_off_diagonal(records, rule="null_dispersion", materiality_threshold=1e6)
    )
    assert row["classification"] == "not_material"


def test_unknown_rule_is_refused():
    records = _records(mode_shape=0.023, anchor_shape=0.006, anchor_shape_spread=0.001)
    with pytest.raises(Phase4SummaryError, match="Unknown materiality rule"):
        localize_off_diagonal(records, rule="whatever")
