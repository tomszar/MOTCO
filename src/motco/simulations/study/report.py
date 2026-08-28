"""Build paper-ready tables and figures from study summaries."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from motco.simulations.grid import (
    SimulationReplicateResult,
    SimulationSummaryResult,
)
from motco.simulations.study.config import Phase4GateConfig
from motco.simulations.study.phase4 import (
    CHECKPOINT_ORDER as _CHECKPOINT_ORDER,
)
from motco.simulations.study.phase4 import (
    MEASUREMENT_SPACES as _MEASUREMENT_SPACES,
)
from motco.simulations.study.phase4 import (
    Phase4GateDecision,
    build_operating_frame,
    evaluate_phase4_gate,
    localize_off_diagonal,
    summarize_attribution,
    summarize_pls_selection,
    summarize_realized_geometry,
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


@dataclass(frozen=True)
class Phase4ReportFrames:
    """Phase 4 structured outputs, all traceable to the merged records.

    Every claim a findings report makes should be readable out of one of these
    tables; they are the primary output and the prose is secondary.
    """

    operating: pd.DataFrame
    geometry: pd.DataFrame
    pls_selection: pd.DataFrame
    attribution: pd.DataFrame
    localization: pd.DataFrame
    gate: pd.DataFrame
    decision: Phase4GateDecision


def build_phase4_frames(
    gate: Phase4GateConfig,
    summaries: Sequence[SimulationSummaryResult],
    records: Sequence[SimulationReplicateResult],
    *,
    expected_units: int | None = None,
) -> Phase4ReportFrames:
    """Build every Phase 4 structured output plus the gate decision."""

    decision = evaluate_phase4_gate(gate, summaries, records, expected_units=expected_units)
    return Phase4ReportFrames(
        operating=build_operating_frame(summaries, records),
        geometry=summarize_realized_geometry(records),
        pls_selection=summarize_pls_selection(records),
        attribution=summarize_attribution(records),
        localization=localize_off_diagonal(records),
        gate=decision.to_frame(),
        decision=decision,
    )


def write_phase4_report(frames: Phase4ReportFrames, out_dir: Path) -> dict[str, Path]:
    """Write the Phase 4 tables as CSV and the gate decision as JSON."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "phase4_operating": out_dir / "phase4_operating.csv",
        "phase4_geometry": out_dir / "phase4_geometry.csv",
        "phase4_pls_selection": out_dir / "phase4_pls_selection.csv",
        "phase4_attribution": out_dir / "phase4_attribution.csv",
        "phase4_localization": out_dir / "phase4_localization.csv",
        "phase4_gate": out_dir / "phase4_gate.csv",
    }
    frames.operating.to_csv(paths["phase4_operating"], index=False)
    frames.geometry.to_csv(paths["phase4_geometry"], index=False)
    frames.pls_selection.to_csv(paths["phase4_pls_selection"], index=False)
    frames.attribution.to_csv(paths["phase4_attribution"], index=False)
    frames.localization.to_csv(paths["phase4_localization"], index=False)
    frames.gate.to_csv(paths["phase4_gate"], index=False)

    decision_path = out_dir / "phase4_gate_decision.json"
    decision_path.write_text(
        json.dumps(
            {
                "decision": frames.decision.decision,
                "rationale": frames.decision.rationale,
                "confirmation_runs": list(frames.decision.confirmation_runs),
                "observations": [
                    {
                        "rule": observation.rule,
                        "kind": observation.kind,
                        "trajectory_mode": observation.trajectory_mode,
                        "statistic": observation.statistic,
                        "cell_id": observation.cell_id,
                        "effect_size": observation.effect_size,
                        "met": observation.met,
                        "detail": observation.detail,
                        "observations": observation.observations,
                    }
                    for observation in frames.decision.observations
                ],
            },
            indent=2,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    paths["phase4_gate_decision"] = decision_path
    return paths


def _json_default(value: object) -> object:
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        item = value.item()
        return None if isinstance(item, float) and not np.isfinite(item) else item
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


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


def render_geometry_checkpoints(frame: pd.DataFrame, out_path: Path) -> Path:
    """Plot each statistic's realized geometry across checkpoints, by mode.

    Every checkpoint is drawn on its own axis with its measurement space in the
    label, because a standardized-feature distance and a PLS-latent distance are
    not on one scale and must not be read off a shared axis.
    """

    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joint = (
        frame[(frame["scope"] == "joint") & (frame["statistic"].isin(("delta", "angle", "shape")))]
        if not frame.empty
        else frame
    )
    if joint.empty:
        return _empty_figure(out_path, "Realized geometry (empty)")

    checkpoints = [c for c in _CHECKPOINT_ORDER if c in set(joint["checkpoint"])]
    statistics = [s for s in ("delta", "angle", "shape") if s in set(joint["statistic"])]
    fig, axes = plt.subplots(
        len(statistics),
        len(checkpoints),
        figsize=(3.2 * len(checkpoints), 2.8 * len(statistics)),
        squeeze=False,
    )
    for row, statistic in enumerate(statistics):
        for column, checkpoint in enumerate(checkpoints):
            ax = axes[row][column]
            block = joint[(joint["statistic"] == statistic) & (joint["checkpoint"] == checkpoint)]
            for mode in sorted(block["trajectory_mode"].dropna().unique().tolist()):
                sub = block[block["trajectory_mode"] == mode].sort_values("effect_size")
                if sub.empty:
                    continue
                ax.plot(
                    sub["effect_size"].to_numpy(dtype=float),
                    sub["mean"].to_numpy(dtype=float),
                    marker="o",
                    label=mode,
                )
            space = _MEASUREMENT_SPACES.get(checkpoint, "unknown")
            if row == 0:
                ax.set_title(f"{checkpoint}\n({space})", fontsize=8)
            if column == 0:
                ax.set_ylabel(statistic)
            ax.set_xlabel("effect size", fontsize=8)
            ax.grid(True, alpha=0.3)
    axes[0][-1].legend(fontsize=7)
    fig.suptitle("Realized geometry by checkpoint (axes are per-space, not comparable)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_selected_components(frame: pd.DataFrame, out_path: Path) -> Path:
    """Plot the distribution of selected PLS component counts per cell."""

    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    block = frame[frame["n_selected_lv"] > 0] if not frame.empty else frame
    if block.empty:
        return _empty_figure(out_path, "Selected PLS components (empty)")

    block = block.sort_values(["trajectory_mode", "effect_size"])
    labels = [
        f"{row.trajectory_mode}@{row.effect_size}" for row in block.itertuples(index=False)
    ]
    x = np.arange(len(block))
    means = block["selected_lv_mean"].to_numpy(dtype=float)
    lows = means - block["selected_lv_min"].to_numpy(dtype=float)
    highs = block["selected_lv_max"].to_numpy(dtype=float) - means
    fig, ax = plt.subplots(figsize=(max(6.0, 0.5 * len(block)), 3.6))
    ax.errorbar(x, means, yerr=np.vstack([lows, highs]), fmt="o", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("selected latent variables")
    ax.set_title("Selected PLS components (mean with observed min/max)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_attribution_stability(frame: pd.DataFrame, out_path: Path) -> Path:
    """Plot cross-replicate top-k agreement and bootstrap sign stability."""

    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    block = frame[frame["component"] == "observed"] if not frame.empty else frame
    if block.empty:
        return _empty_figure(out_path, "Attribution stability (empty)")

    block = block.sort_values(["trajectory_mode", "effect_size", "transition_id"])
    labels = [
        f"{row.trajectory_mode}@{row.effect_size}\n{row.transition_id}"
        for row in block.itertuples(index=False)
    ]
    x = np.arange(len(block))
    fig, ax = plt.subplots(figsize=(max(6.0, 0.7 * len(block)), 3.6))
    for column, label in (
        ("top_k_jaccard", "cross-replicate top-k Jaccard"),
        ("sign_agreement", "cross-replicate sign agreement"),
        ("bootstrap_sign_stability_mean", "bootstrap sign stability"),
    ):
        if column not in block.columns:
            continue
        ax.plot(x, block[column].to_numpy(dtype=float), marker="o", label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("agreement")
    ax.set_title("Attribution stability (observed component)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_phase4_figures(frames: Phase4ReportFrames, out_dir: Path) -> dict[str, Path]:
    """Render every Phase 4 figure alongside the existing study outputs."""

    out_dir = Path(out_dir)
    return {
        "phase4_geometry_plot": render_geometry_checkpoints(
            frames.geometry, out_dir / "phase4_geometry.png"
        ),
        "phase4_selected_components_plot": render_selected_components(
            frames.pls_selection, out_dir / "phase4_selected_components.png"
        ),
        "phase4_attribution_stability_plot": render_attribution_stability(
            frames.attribution, out_dir / "phase4_attribution_stability.png"
        ),
    }


def _empty_figure(out_path: Path, title: str) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_title(title)
    ax.axis("off")
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
    "Phase4ReportFrames",
    "ReportFrames",
    "StudyReportError",
    "build_phase4_frames",
    "build_power_curves",
    "build_specificity_matrix",
    "build_type_i_table",
    "render_power_curves",
    "render_specificity_matrix",
    "render_attribution_stability",
    "render_geometry_checkpoints",
    "render_phase4_figures",
    "render_selected_components",
    "render_type_i_plot",
    "write_phase4_report",
    "write_report_csvs",
]
