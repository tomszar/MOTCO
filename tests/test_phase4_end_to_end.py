"""End-to-end Phase 4 workflow: shards → merge → completeness → report → gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from motco.simulations.grid import read_replicate_results
from motco.simulations.study.config import load_study_config
from motco.simulations.study.enumerate import enumerate_study
from motco.simulations.study.merge import discover_shard_paths, merge_shards
from motco.simulations.study.report import (
    build_phase4_frames,
    render_phase4_figures,
    write_phase4_report,
)
from motco.simulations.study.sharding import run_shard
from motco.simulations.study.summary import summarize_study

pytestmark = pytest.mark.slow

CONFIG = Path("examples/trajectory_power_study/phase4_smoke.json")


@pytest.fixture(scope="module")
def phase4_run(tmp_path_factory) -> dict:
    """Run the committed Phase 4 smoke config end to end, R-free."""

    out_dir = tmp_path_factory.mktemp("phase4")
    config = load_study_config(CONFIG)
    grid = enumerate_study(config)
    for shard_index in range(2):
        run_shard(grid, shard_index=shard_index, n_shards=2, out_dir=out_dir, error_policy="record")

    shards = discover_shard_paths(out_dir)
    merged_path = out_dir / "merged.jsonl"
    records = merge_shards(shards, out_path=merged_path)
    return {"config": config, "grid": grid, "out_dir": out_dir, "records": records}


def test_shards_cover_every_expected_work_unit(phase4_run) -> None:
    expected = sum(cell.n_replicates for cell in phase4_run["grid"].cells)
    records = phase4_run["records"]
    assert len(records) == expected
    assert all(record.status == "completed" for record in records)
    assert len({(r.cell_id, r.replicate_index) for r in records}) == expected


def test_merged_records_round_trip(phase4_run) -> None:
    reloaded = read_replicate_results(phase4_run["out_dir"] / "merged.jsonl")
    assert reloaded == phase4_run["records"]


def test_every_pls_record_carries_integration_and_geometry(phase4_run) -> None:
    for record in phase4_run["records"]:
        assert record.integration_metadata["integration_method"] == "pls"
        assert record.integration_metadata["selected_lv"] >= 1
        assert record.realized_geometry["checkpoints"]["pls_latent"]["joint"]["delta"] is not None


def test_only_selected_orientation_cells_carry_attribution(phase4_run) -> None:
    computed = [r for r in phase4_run["records"] if r.attribution_status == "computed"]
    assert computed, "the smoke config selects at least one orientation cell"
    for record in computed:
        assert record.cell_metadata["trajectory_mode"] == "orientation"
        assert float(record.cell_metadata["effect_size"]) > 0.0
        diagnostics = record.attribution_diagnostics
        assert diagnostics["transitions"]
        assert diagnostics["top_features"]
        assert diagnostics["model"]["selected_components"] == record.integration_metadata["selected_lv"]
        assert diagnostics["units"]["methylation_units"] == "mvalue"

    not_requested = [r for r in phase4_run["records"] if r.attribution_status == "not_requested"]
    assert not_requested
    assert all(r.attribution_diagnostics.get("status") == "not_requested" for r in not_requested)


def test_matched_seeds_pair_primary_cells_in_the_run(phase4_run) -> None:
    by_index: dict[int, set[int]] = {}
    for record in phase4_run["records"]:
        if record.phase != "power_primary":
            continue
        by_index.setdefault(record.replicate_index, set()).add(record.generator_seed)
    assert by_index
    assert all(len(seeds) == 1 for seeds in by_index.values())
    assert len({next(iter(s)) for s in by_index.values()}) == len(by_index)


def test_report_writes_every_phase4_output_and_a_gate_decision(phase4_run) -> None:
    config = phase4_run["config"]
    records = phase4_run["records"]
    summaries = summarize_study(records, alpha=config.alpha)
    expected_units = sum(cell.n_replicates for cell in phase4_run["grid"].cells)
    frames = build_phase4_frames(
        config.acceptance.gate, summaries, records, expected_units=expected_units
    )

    report_dir = phase4_run["out_dir"] / "report"
    paths = {**write_phase4_report(frames, report_dir), **render_phase4_figures(frames, report_dir)}
    for key, path in paths.items():
        assert path.exists() and path.stat().st_size > 0, key

    assert frames.decision.decision in {"proceed", "hold", "indeterminate"}
    payload = json.loads((report_dir / "phase4_gate_decision.json").read_text())
    assert payload["decision"] == frames.decision.decision
    kinds = {observation["kind"] for observation in payload["observations"]}
    assert {"type_i_inflation", "power", "control", "descriptive", "completeness"} <= kinds

    assert not frames.geometry.empty
    assert not frames.pls_selection.empty
    assert not frames.attribution.empty
    assert not frames.operating.empty
    # Every mode's zero point comes from the one shared anchor cell.
    zero = frames.operating[frames.operating["effect_size"] == 0.0]
    assert zero["from_shared_anchor"].all()
    assert zero["cell_id"].nunique() == 1
