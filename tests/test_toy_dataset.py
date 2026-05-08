from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

TOY_DIR = Path("examples/data/toy")
TOY_FILES = {
    "methylation.csv",
    "expression.csv",
    "proteomics.csv",
    "metadata.csv",
    "model_full.csv",
    "model_reduced.csv",
    "ls_means.csv",
    "contrast.json",
    "truth.json",
}


def test_toy_truth_records_canonical_generation_parameters() -> None:
    truth = json.loads((TOY_DIR / "truth.json").read_text())
    n_features = truth["intersim_metadata"]["n_features"]

    assert truth["seed"] == 42
    assert truth["trajectory_mode"] == "orientation"
    assert truth["group_effect_size"] == 1.0
    assert truth["intersim_metadata"]["n_sample"] == 90
    assert truth["intersim_metadata"]["delta_methyl"] == pytest.approx(0.10)
    assert truth["intersim_metadata"]["delta_expr"] == pytest.approx(0.10)
    assert truth["intersim_metadata"]["delta_protein"] == pytest.approx(0.10)

    for layer, features in truth["affected_features"].items():
        expected = round(n_features[layer] * 0.1)
        assert len(features) == pytest.approx(expected, abs=1)


def test_toy_files_are_present_parseable_and_row_aligned() -> None:
    missing = [name for name in TOY_FILES if not (TOY_DIR / name).exists()]
    assert missing == []

    metadata = pd.read_csv(TOY_DIR / "metadata.csv")
    omics = {
        "methylation": pd.read_csv(TOY_DIR / "methylation.csv"),
        "expression": pd.read_csv(TOY_DIR / "expression.csv"),
        "proteomics": pd.read_csv(TOY_DIR / "proteomics.csv"),
    }

    assert list(metadata.columns) == ["sample_id", "group", "stage", "cluster"]
    assert metadata.shape[0] == 90
    for frame in omics.values():
        assert frame.shape[0] == metadata.shape[0]
        assert not frame.empty

    json.loads((TOY_DIR / "contrast.json").read_text())
    json.loads((TOY_DIR / "truth.json").read_text())
    for filename in ("model_full.csv", "model_reduced.csv", "ls_means.csv"):
        assert not pd.read_csv(TOY_DIR / filename).empty
