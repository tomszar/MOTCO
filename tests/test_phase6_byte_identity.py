"""Absent ``group_stage_sizes`` and ``integration_params.layers`` reproduce pre-change output.

``tests/data/phase6_prechange_fixture.json`` was captured by
``tests/phase6_fixture_capture.py`` before either key existed. Integer, string,
and structural content must match exactly; floats are compared at a tight
relative tolerance so the fixture survives BLAS differences across machines.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from tests.phase6_fixture_capture import (
    DATASET_PARAMS,
    EVALUATIONS,
    dataset_summary,
    evaluation_summary,
)

FIXTURE = json.loads(
    (Path(__file__).resolve().parent / "data" / "phase6_prechange_fixture.json").read_text(encoding="utf-8")
)


def _assert_close(actual: Any, expected: Any, path: str = "") -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict), path
        assert sorted(actual) == sorted(expected), path
        for key in expected:
            _assert_close(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), path
        for index, (a, e) in enumerate(zip(actual, expected, strict=True)):
            _assert_close(a, e, f"{path}[{index}]")
    elif isinstance(expected, float) and not isinstance(actual, bool):
        assert isinstance(actual, float | int), path
        assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12), (path, actual, expected)
    else:
        assert actual == expected, path


@pytest.mark.parametrize("name", sorted(DATASET_PARAMS))
def test_absent_size_table_dataset_and_truth_match_pre_change(name: str) -> None:
    _assert_close(json.loads(json.dumps(dataset_summary(DATASET_PARAMS[name]))), FIXTURE["datasets"][name])


@pytest.mark.parametrize("name", sorted(EVALUATIONS))
def test_absent_layer_selection_matches_pre_change(name: str) -> None:
    _assert_close(json.loads(json.dumps(evaluation_summary(name))), FIXTURE["evaluations"][name])


@pytest.mark.parametrize("name", sorted(EVALUATIONS))
def test_explicit_all_layer_selection_equals_absent_selection(name: str, monkeypatch) -> None:
    # The absent selection equals the fixture (test above), so the fixture
    # stands in for it here instead of re-running the evaluation.
    params = EVALUATIONS[name]
    explicit = replace(
        params,
        integration_params={
            **dict(params.integration_params),
            "layers": ["proteomics", "methylation", "expression"],
        },
    )
    monkeypatch.setitem(EVALUATIONS, name, explicit)
    selected = evaluation_summary(name)
    # The declared selection is recorded; everything else is identical.
    recorded = selected["latent_matrix_metadata"]["integration_params"].pop("layers")
    assert recorded == ["methylation", "expression", "proteomics"]
    _assert_close(json.loads(json.dumps(selected)), FIXTURE["evaluations"][name])
