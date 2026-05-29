"""End-to-end smoke test for the trajectory power study pipeline.

Uses a tiny config and a mock evaluator to validate runner → merge → report
without invoking the real InterSIM + RRPP pipeline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from motco.simulations import (
    InterSIMParams,
    SemiSyntheticTrajectoryParams,
    SimulationEvaluationParams,
    SimulationEvaluationResult,
)
from motco.simulations.study import (
    AcceptanceTargets,
    PowerMonotonicityTarget,
    StudyConfig,
    TypeIControlTarget,
    dump_study_config,
    enumerate_study,
)
from motco.simulations.study.merge import discover_shard_paths, merge_shards
from motco.simulations.study.report import (
    ReportFrames,
    build_power_curves,
    build_specificity_matrix,
    build_type_i_table,
    write_report_csvs,
)
from motco.simulations.study.sharding import run_shard
from motco.simulations.study.summary import (
    summarize_combined_rule,
    summarize_study,
)
from motco.simulations.study.targets import evaluate_targets, write_target_report


def _smoke_config() -> StudyConfig:
    return StudyConfig(
        intersim=InterSIMParams(seed=1, n_sample=20),
        generator=SemiSyntheticTrajectoryParams(seed=2, trajectory_mode="magnitude", group_effect_size=0.1),
        evaluation=SimulationEvaluationParams(integration_method="concat", permutations=0, seed=3),
        trajectory_modes=("magnitude",),
        effect_sizes=(0.1, 0.5),
        n_replicates=2,
        base_seed=42,
        alpha=0.05,
        acceptance=AcceptanceTargets(
            type_i=(TypeIControlTarget(alpha=0.05, se_tolerance=2.0),),
            power=(PowerMonotonicityTarget(trajectory_mode="magnitude", statistic="delta", min_power_at_top=0.3),),
        ),
    )


def _mock_evaluator(intersim_params, generator_params, evaluation_params) -> SimulationEvaluationResult:
    """A mock evaluator: nulls reject ~alpha, power cells reject more as effect grows."""

    rng = np.random.default_rng(generator_params.seed)
    matrix = np.zeros((2, 2), dtype=float)
    mode = generator_params.trajectory_mode
    effect = generator_params.group_effect_size
    if mode == "none" or effect == 0.0:
        delta_p = float(rng.uniform(0.0, 1.0))
        angle_p = float(rng.uniform(0.0, 1.0))
    else:
        # higher effect → smaller p-values on average
        bias = max(0.0, 0.6 - 1.2 * effect)
        delta_p = float(np.clip(rng.uniform(0.0, 1.0) - 0.5 + bias, 0.0, 1.0))
        angle_p = float(np.clip(rng.uniform(0.0, 1.0) - 0.3 + bias, 0.0, 1.0))
    return SimulationEvaluationResult(
        observed_deltas=matrix,
        observed_angles=matrix,
        observed_shapes=matrix,
        pair_statistics={"delta": 1.0, "angle": 2.0, "shape": float("nan")},
        p_values={"delta": delta_p, "angle": angle_p},
        latent_matrix_metadata={"integration_method": "concat"},
        truth_metadata={"trajectory_mode": mode, "effect_size": effect},
        runtime_metadata={"runtime_seconds": 0.01},
        evaluation_params=evaluation_params,
        group_levels=["A", "B"],
        stage_levels=["0", "1"],
        contrast=[[0, 1], [2, 3]],
    )


def test_end_to_end_smoke(tmp_path: Path) -> None:
    config = _smoke_config()
    config_path = tmp_path / "smoke.json"
    dump_study_config(config, config_path)

    grid = enumerate_study(config)
    n_shards = 3
    for shard_index in range(n_shards):
        run_shard(
            grid,
            shard_index=shard_index,
            n_shards=n_shards,
            out_dir=tmp_path,
            evaluator=_mock_evaluator,
        )

    shards = discover_shard_paths(tmp_path)
    assert len(shards) == n_shards
    merged_path = tmp_path / "merged.jsonl"
    records = merge_shards(shards, out_path=merged_path)
    assert merged_path.exists()
    total_units = sum(c.n_replicates for c in grid.cells)
    assert len(records) == total_units

    per_stat = summarize_study(records, alpha=config.alpha)
    combined = summarize_combined_rule(records, alpha=config.alpha)
    frames = ReportFrames(
        specificity_matrix=build_specificity_matrix(per_stat, records),
        power_curves=build_power_curves(per_stat, records),
        type_i_table=build_type_i_table(per_stat, combined, records),
    )
    report_dir = tmp_path / "report"
    csv_paths = write_report_csvs(frames, report_dir)
    for path in csv_paths.values():
        assert path.exists()
        assert pd.read_csv(path) is not None

    evaluations = evaluate_targets(config.acceptance, per_stat, records)
    paths = write_target_report(evaluations, report_dir)
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert any(ev["target_kind"] == "type_i_control" for ev in payload)


def test_run_study_shard_script_invocation(tmp_path: Path, monkeypatch) -> None:
    """The runner script wires through to run_shard and emits a JSONL."""

    config = _smoke_config()
    config_path = tmp_path / "smoke.json"
    dump_study_config(config, config_path)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    try:
        import run_study_shard  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)

    with patch("run_study_shard.run_shard") as mock_run:
        mock_run.return_value = []
        rc = run_study_shard.main(
            [
                "--config",
                str(config_path),
                "--out-dir",
                str(tmp_path / "shards"),
                "--shard-index",
                "0",
                "--n-shards",
                "1",
                "--n-jobs",
                "1",
            ]
        )
    assert rc == 0
    mock_run.assert_called_once()


def test_motco_study_script_merge_and_report(tmp_path: Path) -> None:
    """The interactive CLI runs merge then report end-to-end."""

    config = _smoke_config()
    config_path = tmp_path / "smoke.json"
    dump_study_config(config, config_path)
    grid = enumerate_study(config)
    for shard_index in range(2):
        run_shard(
            grid,
            shard_index=shard_index,
            n_shards=2,
            out_dir=tmp_path,
            evaluator=_mock_evaluator,
        )

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    try:
        import motco_study  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)

    assert motco_study.main(["merge", "--out-dir", str(tmp_path)]) == 0
    assert (tmp_path / "merged.jsonl").exists()

    assert (
        motco_study.main(
            ["report", "--config", str(config_path), "--out-dir", str(tmp_path)]
        )
        == 0
    )
    report_dir = tmp_path / "report"
    for name in (
        "specificity_matrix.csv",
        "power_curves.csv",
        "type_i_table.csv",
        "specificity_matrix.png",
        "power_curves.png",
        "type_i.png",
        "acceptance_report.csv",
        "acceptance_report.json",
    ):
        assert (report_dir / name).exists(), f"missing {name}"
