"""Summarize merged replicate results: per-statistic and combined-rule."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from motco.simulations.grid import (
    SimulationReplicateResult,
    SimulationSummaryResult,
    summarize_rejection_rates,
)
from motco.simulations.study.enumerate import NEGATIVE_CONTROL_MODES

DEFAULT_STATISTICS: tuple[str, ...] = ("delta", "angle", "shape")


@dataclass(frozen=True)
class CombinedRuleSummary:
    """Combined-rule (any-statistic) Type I summary for a null cell."""

    cell_id: str
    phase: str
    alpha: float
    completed_replicates: int
    available_replicates: int
    rejected_replicates: int
    rejection_rate: float | None
    monte_carlo_se: float | None
    statistics: tuple[str, ...]


def summarize_study(
    records: Sequence[SimulationReplicateResult],
    *,
    alpha: float = 0.05,
    statistics: Sequence[str] = DEFAULT_STATISTICS,
) -> list[SimulationSummaryResult]:
    """Per-statistic rejection rates with Monte Carlo SE."""

    return summarize_rejection_rates(records, alpha=alpha, statistics=statistics)


def summarize_combined_rule(
    records: Sequence[SimulationReplicateResult],
    *,
    alpha: float = 0.05,
    statistics: Sequence[str] = DEFAULT_STATISTICS,
    null_cell_filter: Callable[[SimulationReplicateResult], bool] | None = None,
) -> list[CombinedRuleSummary]:
    """Compute the combined-rule (reject-if-any) Type I rate per null cell.

    A replicate counts as a rejection if any *available* statistic has p < alpha.
    A replicate counts as *available* if at least one of the requested statistics
    has a finite p-value.

    By default null cells are identified as those whose phase starts with
    ``type_i_`` or whose ``cell_metadata['trajectory_mode']`` is a negative
    control (``none`` or ``translation``).
    """

    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1.")

    is_null = null_cell_filter or _default_null_filter
    by_cell: dict[tuple[str, str], list[SimulationReplicateResult]] = {}
    for record in records:
        if record.status != "completed":
            continue
        if not is_null(record):
            continue
        by_cell.setdefault((record.cell_id, record.phase), []).append(record)

    summaries: list[CombinedRuleSummary] = []
    for (cell_id, phase), group_records in sorted(by_cell.items()):
        available = 0
        rejected = 0
        for record in group_records:
            p_values = [record.p_values.get(stat) for stat in statistics]
            finite = [float(p) for p in p_values if p is not None and math.isfinite(float(p))]
            if not finite:
                continue
            available += 1
            if any(p < alpha for p in finite):
                rejected += 1
        rate = rejected / available if available else None
        se = math.sqrt(rate * (1.0 - rate) / available) if rate is not None else None
        summaries.append(
            CombinedRuleSummary(
                cell_id=cell_id,
                phase=phase,
                alpha=alpha,
                completed_replicates=len(group_records),
                available_replicates=available,
                rejected_replicates=rejected,
                rejection_rate=rate,
                monte_carlo_se=se,
                statistics=tuple(statistics),
            )
        )
    return summaries


def _default_null_filter(record: SimulationReplicateResult) -> bool:
    if record.phase.startswith("type_i_"):
        return True
    mode = record.cell_metadata.get("trajectory_mode")
    if mode in NEGATIVE_CONTROL_MODES:
        return True
    return False


__all__ = [
    "CombinedRuleSummary",
    "DEFAULT_STATISTICS",
    "summarize_combined_rule",
    "summarize_study",
]
