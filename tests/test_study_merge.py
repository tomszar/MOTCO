from __future__ import annotations

import pytest

from motco.simulations import SimulationReplicateResult, append_replicate_results
from motco.simulations.study.merge import StudyMergeError, discover_shard_paths, merge_shards


def _record(cell_id: str, replicate_index: int, signature: str, status: str = "completed") -> SimulationReplicateResult:
    return SimulationReplicateResult(
        cell_id=cell_id,
        phase="type_i_baseline",
        replicate_index=replicate_index,
        replicate_seed=replicate_index,
        generator_seed=replicate_index,
        evaluation_seed=replicate_index,
        parameter_signature=signature,
        status=status,  # type: ignore[arg-type]
    )


def test_merge_combines_shards_and_deduplicates(tmp_path) -> None:
    shard_a = tmp_path / "shard_0.jsonl"
    shard_b = tmp_path / "shard_1.jsonl"
    append_replicate_results(shard_a, [_record("cell-a", 0, "sig-a"), _record("cell-b", 0, "sig-b")])
    append_replicate_results(shard_b, [_record("cell-c", 0, "sig-c"), _record("cell-a", 1, "sig-a")])

    merged = merge_shards([shard_a, shard_b], out_path=tmp_path / "merged.jsonl")
    keys = {(r.cell_id, r.replicate_index) for r in merged}
    assert keys == {("cell-a", 0), ("cell-a", 1), ("cell-b", 0), ("cell-c", 0)}
    assert len(merged) == 4


def test_merge_raises_on_signature_mismatch(tmp_path) -> None:
    shard_a = tmp_path / "shard_0.jsonl"
    shard_b = tmp_path / "shard_1.jsonl"
    append_replicate_results(shard_a, [_record("cell-a", 0, "sig-x")])
    append_replicate_results(shard_b, [_record("cell-a", 0, "sig-y")])

    with pytest.raises(StudyMergeError, match="Inconsistent parameter signatures"):
        merge_shards([shard_a, shard_b])


def test_merge_prefers_completed_over_failed(tmp_path) -> None:
    shard_a = tmp_path / "shard_0.jsonl"
    shard_b = tmp_path / "shard_1.jsonl"
    append_replicate_results(shard_a, [_record("cell-a", 0, "sig-a", status="failed")])
    append_replicate_results(shard_b, [_record("cell-a", 0, "sig-a", status="completed")])

    merged = merge_shards([shard_a, shard_b])
    assert len(merged) == 1
    assert merged[0].status == "completed"


def test_discover_shard_paths(tmp_path) -> None:
    (tmp_path / "shard_0.jsonl").write_text("")
    (tmp_path / "shard_2.jsonl").write_text("")
    (tmp_path / "shard_10.jsonl").write_text("")
    (tmp_path / "ignored.jsonl").write_text("")
    paths = discover_shard_paths(tmp_path)
    assert [p.name for p in paths] == ["shard_0.jsonl", "shard_10.jsonl", "shard_2.jsonl"]
