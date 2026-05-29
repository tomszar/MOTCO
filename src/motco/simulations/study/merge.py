"""Merge per-shard JSONL outputs into a single deduplicated result set."""

from __future__ import annotations

from pathlib import Path

from motco.simulations.grid import (
    SimulationReplicateResult,
    append_replicate_results,
    read_replicate_results,
)


class StudyMergeError(ValueError):
    """Raised when shard outputs cannot be merged consistently."""


def merge_shards(
    shard_paths: list[Path],
    *,
    out_path: Path | None = None,
) -> list[SimulationReplicateResult]:
    """Read all shard JSONL files, deduplicate by `(cell_id, replicate_index)`.

    Raises:
        StudyMergeError: if the same `(cell_id, replicate_index)` appears with
            different parameter signatures across shards.
    """

    by_key: dict[tuple[str, int], SimulationReplicateResult] = {}
    sources: dict[tuple[str, int], Path] = {}
    for path in shard_paths:
        records = read_replicate_results(Path(path))
        for record in records:
            key = (record.cell_id, record.replicate_index)
            existing = by_key.get(key)
            if existing is None:
                by_key[key] = record
                sources[key] = Path(path)
                continue
            if existing.parameter_signature != record.parameter_signature:
                raise StudyMergeError(
                    f"Inconsistent parameter signatures for cell_id={record.cell_id!r}, "
                    f"replicate_index={record.replicate_index} across shards "
                    f"{sources[key]} and {path}."
                )
            # Prefer completed over failed when both exist
            if existing.status == "failed" and record.status == "completed":
                by_key[key] = record
                sources[key] = Path(path)

    merged = [by_key[key] for key in sorted(by_key)]
    if out_path is not None:
        out_path = Path(out_path)
        if out_path.exists():
            out_path.unlink()
        if merged:
            append_replicate_results(out_path, merged)
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.touch()
    return merged


def discover_shard_paths(out_dir: Path) -> list[Path]:
    """Return all shard JSONL files under ``out_dir`` in numeric order."""

    out_dir = Path(out_dir)
    if not out_dir.exists():
        return []
    paths = []
    for path in sorted(out_dir.glob("shard_*.jsonl")):
        try:
            int(path.stem.removeprefix("shard_"))
        except ValueError:
            continue
        paths.append(path)
    return paths


__all__ = ["StudyMergeError", "discover_shard_paths", "merge_shards"]
