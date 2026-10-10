"""Every committed study config keeps its cell ids, signatures, and matched seeds.

The fixture was captured before ``group_stage_sizes`` and
``integration_params.layers`` existed; both keys are absent from every committed
config, so a diff here means an absent key leaked into a hash and a committed
study would no longer resume.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from motco.simulations.grid import derive_replicate_seed, parameter_signature
from motco.simulations.study import enumerate_study, load_study_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "examples" / "trajectory_power_study"
FIXTURE = Path(__file__).resolve().parent / "data" / "committed_config_identity.json"
COMMITTED = json.loads(FIXTURE.read_text(encoding="utf-8"))["configs"]


@pytest.mark.parametrize("name", sorted(COMMITTED))
def test_committed_config_identity_is_unchanged(name: str) -> None:
    grid = enumerate_study(load_study_config(CONFIG_DIR / name))
    actual = {
        cell.cell_id: {
            "signature": parameter_signature(cell),
            "seeds": [derive_replicate_seed(cell, i) for i in range(min(3, cell.n_replicates))],
        }
        for cell in grid.cells
    }
    assert actual == COMMITTED[name]
