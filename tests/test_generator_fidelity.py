"""Swept, replicate-based fidelity of the numpy generator vs InterSIM.

This is the rigorous successor to the single ``delta=0`` realism fixture (see
``test_generator.py``). It compares the numpy generator against committed
InterSIM *summary distributions* (``tests/data/intersim_fidelity_fixture.npz``)
over a ``delta`` x ``p.DMP`` grid, asserting that each numpy statistic's
replicate mean falls inside InterSIM's own central interval.

Criterion (documented): for each statistic, the numpy replicate *mean* must lie
within InterSIM's ``[q2.5, q97.5]`` percentile interval over its ``n_intersim``
replicates. Averaging over numpy replicates removes numpy's Monte-Carlo noise so
the test isolates *systematic* disagreement; the percentile interval absorbs
InterSIM's RNG variability. The fixture is R-free; regenerate it via
``fidelity_intersim.R`` (see ``simulations`` docs).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from motco.simulations.fidelity import (
    FidelityError,
    compare_cell,
    load_fidelity_fixture,
    run_numpy_cell,
    validate_grid,
)
from motco.simulations.reference import load_reference

FIXTURE = Path(__file__).parent / "data" / "intersim_fidelity_fixture.npz"

# A fast representative subset: the degenerate anchor and the strongest effect,
# at the lower DMP density. Enough to guard the baseline + effect-injection +
# coupling paths without the full 6-cell sweep.
FAST_CELLS = [(0.0, 0.2), (2.0, 0.2)]
FAST_N_NUMPY = 8


@pytest.fixture(scope="module")
def fixture():
    return load_fidelity_fixture(FIXTURE)


@pytest.fixture(scope="module")
def reference():
    return load_reference()


def test_load_fidelity_fixture_missing_raises(tmp_path):
    with pytest.raises(FidelityError, match="fidelity fixture not found"):
        load_fidelity_fixture(tmp_path / "absent.npz")


def test_fixture_provenance_records_intersim(fixture):
    assert fixture.provenance["intersim_version"]
    assert fixture.provenance["generation_date"]
    assert len(fixture.cells) == 6
    assert all(len(dist) == 24 for dist in fixture.distributions)


def test_fidelity_fast_subset(fixture, reference):
    """Numpy falls within InterSIM's interval for the fast cell subset."""

    grid = dataclasses.replace(fixture.grid, n_numpy=FAST_N_NUMPY)
    rng = np.random.default_rng(grid.seed)
    for delta, p_dmp in FAST_CELLS:
        idx = fixture.cell_index(delta, p_dmp)
        numpy_dist = run_numpy_cell(delta, p_dmp, grid, reference=reference, rng=rng)
        comparisons = compare_cell(numpy_dist, fixture.distributions[idx])
        failures = [c.statistic for c in comparisons.values() if not c.passed]
        assert not failures, f"cell (delta={delta}, p_dmp={p_dmp}) failed: {failures}"


def test_effect_injection_and_coupling_at_nonzero_delta(fixture, reference):
    """delta>0 exercises effect injection + DMP->DEG->DEP coupling (the delta=0 gap).

    At delta=0 cluster separation and the differential-feature rate are ~0 for
    all omics; at delta>0 they become clearly positive *and* agree with
    InterSIM's distribution across methylation, expression, and protein -- which
    only happens if the cross-omic coupling fires as InterSIM's does.
    """

    grid = dataclasses.replace(fixture.grid, n_numpy=FAST_N_NUMPY)
    rng = np.random.default_rng(grid.seed)

    null = run_numpy_cell(0.0, 0.2, grid, reference=reference, rng=rng)
    effect = run_numpy_cell(2.0, 0.2, grid, reference=reference, rng=rng)

    for omic in ("methylation", "expression", "proteomics"):
        eta2_null = float(np.mean(null[f"{omic}_eta2"]))
        eta2_effect = float(np.mean(effect[f"{omic}_eta2"]))
        rate_effect = float(np.mean(effect[f"{omic}_diff_rate"]))
        # Effect injection lifts separation well clear of the null regime.
        assert eta2_null < 0.05, f"{omic} eta2 not ~0 at delta=0: {eta2_null}"
        assert eta2_effect > eta2_null + 0.05, f"{omic} eta2 did not rise: {eta2_effect}"
        # Coupling propagates differential features to every omic.
        assert rate_effect > 0.0, f"{omic} has no differential features at delta>0"

    # And the lifted values agree with InterSIM, not just with themselves.
    idx = fixture.cell_index(2.0, 0.2)
    comparisons = compare_cell(effect, fixture.distributions[idx])
    for omic in ("methylation", "expression", "proteomics"):
        assert comparisons[f"{omic}_eta2"].passed, f"{omic} eta2 outside InterSIM interval"
        assert comparisons[f"{omic}_diff_rate"].passed, f"{omic} diff_rate outside interval"


@pytest.mark.slow
def test_fidelity_full_grid(fixture, reference):
    """Every statistic in every grid cell falls within InterSIM's interval."""

    results = validate_grid(fixture, reference=reference)
    failures = {
        cell: [c.statistic for c in comps.values() if not c.passed]
        for cell, comps in results.items()
    }
    failures = {cell: stats for cell, stats in failures.items() if stats}
    assert not failures, f"fidelity failures: {failures}"
