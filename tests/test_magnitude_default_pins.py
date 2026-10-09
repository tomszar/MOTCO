"""Historical study configs keep their identity across the ``magnitude_kind`` default flip.

The fixture was captured on the pre-flip code with the configs unedited, so it
records what was actually committed and run. Each config now pins
``"magnitude_kind": "all"``; these tests prove the pin is behavior-neutral.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from motco.simulations.grid import parameter_signature
from motco.simulations.study import enumerate_study, load_study_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "examples" / "trajectory_power_study"
FIXTURE = Path(__file__).resolve().parent / "data" / "historical_config_signatures.json"
HISTORICAL = json.loads(FIXTURE.read_text(encoding="utf-8"))["configs"]


def test_fixture_covers_the_ten_historical_configs() -> None:
    assert len(HISTORICAL) == 10
    assert all(HISTORICAL.values())


@pytest.mark.parametrize("name", sorted(HISTORICAL))
def test_historical_config_signatures_are_unchanged(name: str) -> None:
    config = load_study_config(CONFIG_DIR / name)
    assert config.generator.magnitude_kind == "all"

    grid = enumerate_study(config)
    actual = {cell.cell_id: parameter_signature(cell) for cell in grid.cells}
    expected = HISTORICAL[name]
    assert set(actual) == set(expected)
    assert actual == expected


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.json")), ids=lambda p: p.name)
def test_every_committed_config_names_its_magnitude_kind(path: Path) -> None:
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert "magnitude_kind" in raw["generator"], (
        f"{path.name} must choose generator.magnitude_kind deliberately; "
        "do not copy the historical 'all' pin without reason."
    )
