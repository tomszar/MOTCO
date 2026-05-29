"""Build paper-ready tables and figures from study summaries."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from motco.simulations.grid import (
    SimulationReplicateResult,
    SimulationSummaryResult,
)
from motco.simulations.study.summary import CombinedRuleSummary


class StudyReportError(ValueError):
    """Raised when summaries cannot be turned into a report."""


@dataclass(frozen=True)
class ReportFrames:
    """Paper-ready tables built from study summaries."""

    specificity_matrix: pd.DataFrame
    power_curves: pd.DataFrame
    type_i_table: pd.DataFrame


def build_specificity_matrix(
    summaries: Sequence[SimulationSummaryResult],
    records: Sequence[SimulationReplicateResult],
) -> pd.DataFrame:
    """Mode × statistic matrix of rejection rates (rate ± SE) at the top effect size.

    Rows are trajectory modes (from cell metadata), columns are statistics. For modes
    with multiple effect sizes (power cells), the entry for that mode uses the
    *largest* effect size; for negative-control / null modes the entry uses null cells.
    """

    cell_meta = _cell_metadata_index(records)
    rows: list[dict] = []
    for summary in summaries:
        meta = cell_meta.get(summary.cell_id)
        if meta is None:
            continue
        mode = _resolve_mode(meta)
        effect_size = float(meta.get("effect_size", 0.0) or 0.0)
        varied_axis = meta.get("varied_axis")
        if varied_axis is not None:
            continue
        rows.append(
            {
                "trajectory_mode": mode,
                "statistic": summary.statistic,
                "effect_size": effect_size,
                "cell_id": summary.cell_id,
                "phase": summary.phase,
                "rejection_rate": summary.rejection_rate,
                "monte_carlo_se": summary.monte_carlo_se,
                "available_replicates": summary.available_replicates,
                "completed_replicates": summary.completed_replicates,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "trajectory_mode",
                "statistic",
                "effect_size",
                "rejection_rate",
                "monte_carlo_se",
                "available_replicates",
            ]
        )
    frame = pd.DataFrame(rows)
    # for each (mode, statistic), keep the largest effect_size row
    idx = frame.groupby(["trajectory_mode", "statistic"])["effect_size"].idxmax()
    frame = frame.loc[idx].reset_index(drop=True)
    return frame.sort_values(["trajectory_mode", "statistic"]).reset_index(drop=True)


def build_power_curves(
    summaries: Sequence[SimulationSummaryResult],
    records: Sequence[SimulationReplicateResult],
) -> pd.DataFrame:
    """One row per (mode, statistic, effect_size) over baseline (non-OFAT) cells."""

    cell_meta = _cell_metadata_index(records)
    rows: list[dict] = []
    for summary in summaries:
        meta = cell_meta.get(summary.cell_id)
        if meta is None:
            continue
        if meta.get("varied_axis") is not None:
            continue
        mode = _resolve_mode(meta)
        effect_size = meta.get("effect_size")
        if effect_size is None:
            continue
        rows.append(
            {
                "trajectory_mode": mode,
                "statistic": summary.statistic,
                "effect_size": float(effect_size),
                "cell_id": summary.cell_id,
                "phase": summary.phase,
                "rejection_rate": summary.rejection_rate,
                "monte_carlo_se": summary.monte_carlo_se,
                "available_replicates": summary.available_replicates,
                "completed_replicates": summary.completed_replicates,
            }
        )
    frame = pd.DataFrame(
        rows,
        columns=[
            "trajectory_mode",
            "statistic",
            "effect_size",
            "cell_id",
            "phase",
            "rejection_rate",
            "monte_carlo_se",
            "available_replicates",
            "completed_replicates",
        ],
    )
    if frame.empty:
        return frame
    return frame.sort_values(
        ["trajectory_mode", "statistic", "effect_size"]
    ).reset_index(drop=True)


def build_type_i_table(
    summaries: Sequence[SimulationSummaryResult],
    combined: Sequence[CombinedRuleSummary],
    records: Sequence[SimulationReplicateResult],
) -> pd.DataFrame:
    """Null-cell table: per-statistic + combined-rule rejection rates."""

    cell_meta = _cell_metadata_index(records)
    by_cell: dict[str, dict] = {}
    for summary in summaries:
        meta = cell_meta.get(summary.cell_id)
        if meta is None:
            continue
        if not summary.phase.startswith("type_i_"):
            continue
        entry = by_cell.setdefault(
            summary.cell_id,
            {
                "cell_id": summary.cell_id,
                "phase": summary.phase,
                "trajectory_mode": _resolve_mode(meta),
                "varied_axis": meta.get("varied_axis"),
                "varied_value": meta.get("varied_value"),
                "completed_replicates": summary.completed_replicates,
            },
        )
        entry[f"{summary.statistic}_rate"] = summary.rejection_rate
        entry[f"{summary.statistic}_se"] = summary.monte_carlo_se
        entry[f"{summary.statistic}_available"] = summary.available_replicates

    combined_by_cell = {row.cell_id: row for row in combined}
    for cell_id, entry in by_cell.items():
        crow = combined_by_cell.get(cell_id)
        entry["combined_rate"] = crow.rejection_rate if crow else None
        entry["combined_se"] = crow.monte_carlo_se if crow else None
        entry["combined_available"] = crow.available_replicates if crow else 0

    frame = pd.DataFrame(list(by_cell.values()))
    if frame.empty:
        return frame
    return frame.sort_values(["trajectory_mode", "varied_axis", "varied_value", "cell_id"]).reset_index(drop=True)


def write_report_csvs(
    frames: ReportFrames,
    out_dir: Path,
) -> dict[str, Path]:
    """Write the three report frames as CSVs under ``out_dir``."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "specificity_matrix": out_dir / "specificity_matrix.csv",
        "power_curves": out_dir / "power_curves.csv",
        "type_i_table": out_dir / "type_i_table.csv",
    }
    frames.specificity_matrix.to_csv(paths["specificity_matrix"], index=False)
    frames.power_curves.to_csv(paths["power_curves"], index=False)
    frames.type_i_table.to_csv(paths["type_i_table"], index=False)
    return paths


def render_specificity_matrix(frame: pd.DataFrame, out_path: Path) -> Path:
    """Render the specificity matrix as a heatmap PNG."""

    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.set_title("Specificity matrix (empty)")
        ax.axis("off")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path
    pivot = frame.pivot(index="trajectory_mode", columns="statistic", values="rejection_rate")
    fig, ax = plt.subplots(figsize=(1.2 * pivot.shape[1] + 2, 0.6 * pivot.shape[0] + 1.5))
    im = ax.imshow(pivot.to_numpy(dtype=float), cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(list(pivot.columns), rotation=45, ha="right")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(list(pivot.index))
    ax.set_xlabel("statistic")
    ax.set_ylabel("trajectory_mode")
    ax.set_title("Rejection rate by mode × statistic")
    values = pivot.to_numpy(dtype=float)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = values[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label="rejection rate")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_type_i_plot(
    frame: pd.DataFrame,
    out_path: Path,
    *,
    alpha: float = 0.05,
) -> Path:
    """Plot Type I rates (per statistic + combined) vs alpha."""

    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.set_title("Type I table (empty)")
        ax.axis("off")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path
    statistics = [c[:-5] for c in frame.columns if c.endswith("_rate")]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(frame))
    width = 0.8 / max(len(statistics), 1)
    for i, stat in enumerate(statistics):
        rate_col = f"{stat}_rate"
        se_col = f"{stat}_se"
        rates = frame[rate_col].to_numpy(dtype=float)
        ses = frame[se_col].to_numpy(dtype=float) if se_col in frame.columns else np.zeros_like(rates)
        ax.bar(x + i * width - 0.4 + width / 2, rates, width=width, yerr=ses, label=stat, capsize=2)
    ax.axhline(alpha, color="black", linestyle="--", linewidth=1, label=f"alpha={alpha}")
    labels = [
        f"{str(row.trajectory_mode)}\n({str(row.varied_axis) if row.varied_axis else 'baseline'})"
        for row in frame.itertuples(index=False)
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("rejection rate")
    ax.set_title("Type I rejection rates on null cells")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_power_curves(
    frame: pd.DataFrame,
    out_path: Path,
) -> Path:
    """Render a panel grid of rejection-rate curves vs effect size."""

    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.set_title("Power curves (empty)")
        ax.axis("off")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    modes = sorted(frame["trajectory_mode"].dropna().unique().tolist())
    statistics = sorted(frame["statistic"].dropna().unique().tolist())
    n_modes = len(modes)
    fig, axes = plt.subplots(
        1, n_modes, figsize=(3.4 * n_modes, 3.2), sharey=True, squeeze=False
    )
    for j, mode in enumerate(modes):
        ax = axes[0, j]
        sub = frame[frame["trajectory_mode"] == mode]
        for stat in statistics:
            stat_sub = sub[sub["statistic"] == stat].sort_values("effect_size")
            if stat_sub.empty:
                continue
            ax.errorbar(
                stat_sub["effect_size"].to_numpy(dtype=float),
                stat_sub["rejection_rate"].to_numpy(dtype=float),
                yerr=stat_sub["monte_carlo_se"].to_numpy(dtype=float),
                marker="o",
                label=stat,
                capsize=2,
            )
        ax.set_title(mode)
        ax.set_xlabel("effect size")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.set_ylabel("rejection rate")
        ax.legend(fontsize=8)
    fig.suptitle("Power curves by trajectory mode")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _cell_metadata_index(records: Iterable[SimulationReplicateResult]) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for record in records:
        if record.cell_id in index:
            continue
        index[record.cell_id] = dict(record.cell_metadata)
    return index


def _resolve_mode(meta: dict) -> str:
    mode = meta.get("trajectory_mode")
    if mode is not None:
        return str(mode)
    # baseline Type I cells from enumerate_type_i_grid carry no trajectory_mode metadata;
    # they are always the `none` mode (group_effect_size = 0).
    return "none"


__all__ = [
    "ReportFrames",
    "StudyReportError",
    "build_power_curves",
    "build_specificity_matrix",
    "build_type_i_table",
    "render_power_curves",
    "render_specificity_matrix",
    "render_type_i_plot",
    "write_report_csvs",
]
