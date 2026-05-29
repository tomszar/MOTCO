"""Deterministic sharding of `(cell, replicate)` units for cluster execution."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from motco.simulations.grid import (
    Evaluator,
    SimulationCell,
    SimulationGrid,
    SimulationReplicateResult,
    SimulationRunConfig,
    _completed_index,
    append_replicate_results,
    parameter_signature,
    run_simulation_replicate,
)


class StudyShardError(ValueError):
    """Raised when shard parameters are invalid."""


@dataclass(frozen=True)
class StudyUnit:
    """One `(cell, replicate_index)` work unit."""

    cell: SimulationCell
    replicate_index: int

    @property
    def key(self) -> tuple[str, int]:
        return (self.cell.cell_id, self.replicate_index)


def enumerate_units(grid: SimulationGrid) -> list[StudyUnit]:
    """Return all `(cell, replicate)` units in deterministic order."""

    units: list[StudyUnit] = []
    for cell in sorted(grid.cells, key=lambda c: c.cell_id):
        for replicate_index in range(cell.n_replicates):
            units.append(StudyUnit(cell=cell, replicate_index=replicate_index))
    return units


def partition_unit(cell_id: str, replicate_index: int, *, n_shards: int) -> int:
    """Map a `(cell_id, replicate_index)` unit to a shard index in `[0, n_shards)`.

    The partition is a stable SHA-256 hash modulo `n_shards`, giving the
    same shard assignment across processes regardless of grid order.
    """

    if n_shards <= 0:
        raise StudyShardError(f"n_shards must be positive, got {n_shards}.")
    payload = f"{cell_id}|{replicate_index}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    value = int.from_bytes(digest[:8], "big")
    return value % n_shards


def shard_path(out_dir: Path, shard_index: int) -> Path:
    """Path to the JSONL output file for a given shard."""

    return Path(out_dir) / f"shard_{shard_index}.jsonl"


def run_shard(
    grid: SimulationGrid,
    *,
    shard_index: int,
    n_shards: int,
    out_dir: Path,
    evaluator: Evaluator | None = None,
    error_policy: str = "raise",
    overwrite: bool = False,
) -> list[SimulationReplicateResult]:
    """Execute the units assigned to ``shard_index`` and persist to its JSONL file."""

    if shard_index < 0 or shard_index >= n_shards:
        raise StudyShardError(f"shard_index must be in [0, {n_shards - 1}], got {shard_index}.")
    if error_policy not in {"raise", "record"}:
        raise StudyShardError(f"Unsupported error_policy: {error_policy!r}.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = shard_path(out_dir, shard_index)

    completed = {} if overwrite else _completed_index(path)
    if overwrite and path.exists():
        path.unlink()

    config = SimulationRunConfig(
        output_path=path,
        resume=True,
        overwrite=overwrite,
        error_policy=error_policy,  # type: ignore[arg-type]
    )

    records: list[SimulationReplicateResult] = []
    for unit in enumerate_units(grid):
        cell = unit.cell
        replicate_index = unit.replicate_index
        if partition_unit(cell.cell_id, replicate_index, n_shards=n_shards) != shard_index:
            continue
        signature = parameter_signature(cell)
        existing_signature = completed.get((cell.cell_id, replicate_index))
        if existing_signature is not None:
            if existing_signature != signature and not overwrite:
                raise StudyShardError(
                    f"Existing shard {shard_index} result for {cell.cell_id} replicate "
                    f"{replicate_index} has a different parameter signature. "
                    "Re-run with overwrite=True after deleting the shard file."
                )
            if existing_signature == signature:
                continue
        record = run_simulation_replicate(
            cell,
            replicate_index,
            evaluator=evaluator,
            error_policy=config.error_policy,
        )
        records.append(record)
        append_replicate_results(path, [record])
    return records


__all__ = [
    "StudyShardError",
    "StudyUnit",
    "enumerate_units",
    "partition_unit",
    "run_shard",
    "shard_path",
]
