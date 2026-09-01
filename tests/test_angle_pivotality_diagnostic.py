"""The committed angle-pivotality diagnostic profile: enumeration and a small-scale run."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from motco.simulations.grid import read_replicate_results
from motco.simulations.study.config import load_study_config
from motco.simulations.study.enumerate import enumerate_study
from motco.simulations.study.sharding import run_shard

CONFIG = Path("examples/trajectory_power_study/angle_pivotality_diagnostic.json")
PHASE4_CONFIG = Path("examples/trajectory_power_study/phase4_pilot_100x199.json")

#: The three cells the diagnostic is about: the shortfall, the null control, and
#: the calibrated comparator, each at the top effect size.
DIAGNOSTIC_MODES = {"orientation", "translation", "magnitude"}


def test_diagnostic_profile_enumerates_the_three_cells_plus_the_mandated_controls() -> None:
    """The profile's three power cells, and the two controls the enumerator always adds.

    ``enumerate_study`` unconditionally emits a ``none`` Type I baseline and a
    ``translation`` negative control. They are kept rather than suppressed: the
    cross-replicate counterfactual calibrates its ``z`` against exactly those
    null-control replicates, so they are the diagnostic's reference distribution.
    """

    config = load_study_config(CONFIG)
    grid = enumerate_study(config)

    primary = [cell for cell in grid.cells if cell.phase == "power_primary"]
    controls = [cell for cell in grid.cells if cell.phase == "type_i_baseline"]
    assert len(grid.cells) == 5
    assert {str(cell.generator_params.trajectory_mode) for cell in primary} == DIAGNOSTIC_MODES
    assert all(cell.generator_params.group_effect_size == 1.0 for cell in primary)
    assert all(cell.n_replicates == 100 for cell in grid.cells)
    assert all(cell.evaluation_params.permutations == 199 for cell in grid.cells)
    assert all(cell.evaluation_params.n_jobs == 1 for cell in grid.cells)
    assert all(not cell.evaluation_params.attribution.enabled for cell in grid.cells)
    # The controls: the null baseline and the location-only offset.
    assert {str(cell.generator_params.trajectory_mode) for cell in controls} == {"none", "translation"}


def test_diagnostic_profile_matches_the_phase4_contract() -> None:
    """Only the counts and the diagnostic's own switches may differ from Phase 4."""

    diagnostic = load_study_config(CONFIG)
    phase4 = load_study_config(PHASE4_CONFIG)

    assert diagnostic.generator == phase4.generator
    assert diagnostic.base_seed == phase4.base_seed
    assert diagnostic.alpha == phase4.alpha
    assert diagnostic.n_replicates == phase4.n_replicates
    # Evaluation differs only in the attribution selector, which is off here.
    assert replace(diagnostic.evaluation, attribution=phase4.evaluation.attribution) == phase4.evaluation
    assert diagnostic.evaluation.permutations == phase4.evaluation.permutations == 199
    assert diagnostic.evaluation.integration_method == "pls"
    assert diagnostic.evaluation.integration_params == phase4.evaluation.integration_params
    assert diagnostic.effect_sizes == (1.0,)
    assert not diagnostic.attribution.enabled
    assert diagnostic.metadata["derives_from"] == str(PHASE4_CONFIG)


def test_diagnostic_profile_declares_its_counts_explicitly() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))

    assert payload["n_replicates"] == 100
    assert payload["evaluation"]["permutations"] == 199
    assert payload["trajectory_modes"] == ["magnitude", "orientation", "translation"]
    assert payload["effect_sizes"] == [1.0]


@pytest.fixture(scope="module")
def small_scale_run(tmp_path_factory) -> dict:
    """Run the committed profile at reduced scale through the real study runner."""

    out_dir = tmp_path_factory.mktemp("angle_pivotality")
    config = load_study_config(CONFIG)
    grid = enumerate_study(config)
    reduced = replace(
        grid,
        cells=tuple(
            replace(cell, n_replicates=1, evaluation_params=replace(cell.evaluation_params, permutations=9))
            for cell in grid.cells
        ),
    )
    run_shard(reduced, shard_index=0, n_shards=1, out_dir=out_dir, error_policy="record")
    return {
        "out_dir": out_dir,
        "grid": reduced,
        "records": read_replicate_results(out_dir / "shard_0.jsonl"),
    }


@pytest.mark.slow
def test_diagnostic_profile_runs_end_to_end_and_records_null_summaries(small_scale_run) -> None:
    records = small_scale_run["records"]
    reduced = small_scale_run["grid"]

    assert len(records) == sum(cell.n_replicates for cell in reduced.cells)
    assert all(record.status == "completed" for record in records)
    for record in records:
        summary = record.null_summary
        assert set(summary) == {"delta", "angle", "shape"}
        for statistic, entry in summary.items():
            assert entry["count"] == 9.0, statistic
            assert {"mean", "sd", "q50", "q90", "q95", "q99"} <= set(entry), statistic
        # The pivotality question's core pair is answerable from this one record.
        assert record.pair_statistics["angle"] is not None
        assert summary["angle"]["q95"] > 0.0


@pytest.mark.slow
def test_pivotality_script_produces_tables_from_the_small_scale_records(small_scale_run) -> None:
    """`scripts/angle_null_pivotality.py` runs over the profile's own records."""

    import csv
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "angle_null_pivotality", Path("scripts/angle_null_pivotality.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    report_dir = small_scale_run["out_dir"] / "report"
    exit_code = module.main(
        [
            "--merged",
            str(small_scale_run["out_dir"] / "shard_0.jsonl"),
            "--out-dir",
            str(report_dir),
        ]
    )

    assert exit_code == 0
    for name in (
        "pivotality_association",
        "pivotality_rejection_split",
        "pivotality_standardized",
    ):
        with open(report_dir / f"{name}.csv", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert rows, name
        assert {row["trajectory_mode"] for row in rows} >= DIAGNOSTIC_MODES | {"none"}


@pytest.mark.slow
def test_pivotality_script_refuses_records_without_null_summaries(tmp_path, small_scale_run) -> None:
    """Records predating the null-summary field cannot answer the question."""

    import importlib.util
    import json

    spec = importlib.util.spec_from_file_location(
        "angle_null_pivotality", Path("scripts/angle_null_pivotality.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    legacy = tmp_path / "legacy.jsonl"
    with legacy.open("w", encoding="utf-8") as handle:
        for record in small_scale_run["records"]:
            payload = record.__dict__.copy()
            payload.pop("null_summary")
            handle.write(json.dumps(payload) + "\n")

    assert module.main(["--merged", str(legacy), "--out-dir", str(tmp_path / "report")]) == 2
