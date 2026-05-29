from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from motco.simulations import (
    InterSIMParams,
    SemiSyntheticTrajectoryParams,
    SimulationEvaluationParams,
    SimulationEvaluationResult,
    SimulationGrid,
    make_simulation_cell,
    read_replicate_results,
)
from motco.simulations.study.sharding import (
    StudyShardError,
    enumerate_units,
    partition_unit,
    run_shard,
    shard_path,
)


def _make_grid(n_cells: int, n_replicates: int) -> SimulationGrid:
    cells = []
    for i in range(n_cells):
        cells.append(
            make_simulation_cell(
                phase="type_i_baseline",
                intersim_params=InterSIMParams(seed=i, n_sample=20),
                generator_params=SemiSyntheticTrajectoryParams(seed=i),
                evaluation_params=SimulationEvaluationParams(integration_method="concat", permutations=0, seed=i),
                n_replicates=n_replicates,
                cell_id=f"cell-{i:02d}",
            )
        )
    return SimulationGrid(cells=tuple(cells))


def _fake_result() -> SimulationEvaluationResult:
    matrix = np.zeros((2, 2), dtype=float)
    return SimulationEvaluationResult(
        observed_deltas=matrix,
        observed_angles=matrix,
        observed_shapes=matrix,
        pair_statistics={"delta": 1.0, "angle": 2.0, "shape": float("nan")},
        p_values={"delta": 0.01, "angle": 0.20},
        latent_matrix_metadata={"integration_method": "concat"},
        truth_metadata={},
        runtime_metadata={"runtime_seconds": 0.1},
        evaluation_params=SimulationEvaluationParams(integration_method="concat", permutations=0, seed=1),
        group_levels=["A", "B"],
        stage_levels=["0", "1"],
        contrast=[[0, 1], [2, 3]],
    )


def test_partition_is_exhaustive_and_non_overlapping() -> None:
    grid = _make_grid(n_cells=10, n_replicates=4)
    n_shards = 5
    units = enumerate_units(grid)
    assignments = [partition_unit(u.cell.cell_id, u.replicate_index, n_shards=n_shards) for u in units]
    counts = Counter(assignments)
    assert sum(counts.values()) == len(units)
    assert set(counts.keys()).issubset(set(range(n_shards)))
    # determinism
    again = [partition_unit(u.cell.cell_id, u.replicate_index, n_shards=n_shards) for u in units]
    assert assignments == again


def test_partition_rejects_invalid_n_shards() -> None:
    with pytest.raises(StudyShardError, match="positive"):
        partition_unit("cell", 0, n_shards=0)


def test_run_shard_executes_only_assigned_units(tmp_path) -> None:
    grid = _make_grid(n_cells=4, n_replicates=3)
    n_shards = 3
    calls = 0

    def evaluator(intersim_params, generator_params, evaluation_params) -> SimulationEvaluationResult:
        nonlocal calls
        calls += 1
        return _fake_result()

    expected_per_shard: dict[int, set] = {i: set() for i in range(n_shards)}
    for unit in enumerate_units(grid):
        s = partition_unit(unit.cell.cell_id, unit.replicate_index, n_shards=n_shards)
        expected_per_shard[s].add((unit.cell.cell_id, unit.replicate_index))

    for shard_index in range(n_shards):
        records = run_shard(
            grid,
            shard_index=shard_index,
            n_shards=n_shards,
            out_dir=tmp_path,
            evaluator=evaluator,
        )
        produced = {(r.cell_id, r.replicate_index) for r in records}
        assert produced == expected_per_shard[shard_index]

    total_executed = calls
    assert total_executed == 4 * 3
    union = set()
    for shard_index in range(n_shards):
        loaded = read_replicate_results(shard_path(tmp_path, shard_index))
        union.update((r.cell_id, r.replicate_index) for r in loaded)
    assert union == {(u.cell.cell_id, u.replicate_index) for u in enumerate_units(grid)}


def test_run_shard_resumes_without_duplicating(tmp_path) -> None:
    grid = _make_grid(n_cells=4, n_replicates=2)

    def evaluator(intersim_params, generator_params, evaluation_params) -> SimulationEvaluationResult:
        return _fake_result()

    first = run_shard(grid, shard_index=0, n_shards=2, out_dir=tmp_path, evaluator=evaluator)
    second = run_shard(grid, shard_index=0, n_shards=2, out_dir=tmp_path, evaluator=evaluator)
    assert second == []
    loaded = read_replicate_results(shard_path(tmp_path, 0))
    assert {(r.cell_id, r.replicate_index) for r in loaded} == {
        (r.cell_id, r.replicate_index) for r in first
    }


def test_run_shard_records_failures_with_policy(tmp_path) -> None:
    grid = _make_grid(n_cells=1, n_replicates=1)

    def evaluator(intersim_params, generator_params, evaluation_params) -> SimulationEvaluationResult:
        raise RuntimeError("boom")

    shard_index = partition_unit(grid.cells[0].cell_id, 0, n_shards=1)
    records = run_shard(
        grid,
        shard_index=shard_index,
        n_shards=1,
        out_dir=tmp_path,
        evaluator=evaluator,
        error_policy="record",
    )
    assert len(records) == 1
    assert records[0].status == "failed"
    assert records[0].error_type == "RuntimeError"


def test_run_shard_rejects_invalid_index(tmp_path) -> None:
    grid = _make_grid(n_cells=1, n_replicates=1)
    with pytest.raises(StudyShardError, match="shard_index"):
        run_shard(grid, shard_index=5, n_shards=2, out_dir=tmp_path)
