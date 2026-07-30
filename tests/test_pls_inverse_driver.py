"""Smoke tests for the inverse-PLS study driver."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def test_driver_writes_deterministic_output(tmp_path: Path) -> None:
    script = Path(__file__).parents[1] / "scripts" / "pls_inverse_interpretability.py"
    spec = importlib.util.spec_from_file_location("pls_inverse_interpretability", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = [
        "--seed",
        "3",
        "--n-features",
        "8",
        "--n-samples-per-stage",
        "6",
        "--out-dir",
        str(tmp_path),
    ]
    assert module.main(args) == 0
    first = (tmp_path / "cells.csv").read_text()
    assert module.main(args) == 0
    assert (tmp_path / "cells.csv").read_text() == first
    cells = pd.read_csv(tmp_path / "cells.csv")
    features = pd.read_csv(tmp_path / "features.csv")
    assert cells.shape[0] == 5
    assert features.shape[0] == 40
    assert "Interpretation boundary" in (tmp_path / "report.md").read_text()
