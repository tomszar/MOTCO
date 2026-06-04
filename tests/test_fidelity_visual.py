"""Tests for the qualitative fidelity visuals (fidelity_visual.py).

The real InterSIM visual fixture is *not* committed (it needs InterSIM, via
flake.nix, and the raw matrices are large). These tests therefore exercise the
plumbing R-free with a small *synthetic* InterSIM export: the export/pack/load
round-trip, the matched numpy generation, and the figure builders. The numpy
side uses the committed reference cache, so it is genuine; only the stand-in
"InterSIM" matrices are synthetic, which is enough to smoke-render every figure.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest
from matplotlib.figure import Figure

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from motco.simulations.fidelity import FidelityError
from motco.simulations.fidelity_visual import (
    _FIGURE_BUILDERS,
    VisualData,
    build_visual_fixture_from_export,
    generate_numpy_visual_data,
    load_visual_fixture,
    run_fidelity_visual,
)
from motco.simulations.generator import OMIC_LAYERS
from motco.simulations.reference import load_reference

N_SAMPLE = 24
N_CLUSTER = 2
N_REP_DENSITY = 2
SUBSAMPLE = 80


@pytest.fixture(scope="module")
def reference():
    return load_reference()


def _feat_counts(reference) -> dict[str, int]:
    return {
        "methylation": reference.n_cpg,
        "expression": reference.n_gene,
        "proteomics": reference.n_protein,
    }


def _write_synthetic_export(directory, reference):
    """Write a tiny stand-in for ``fidelity_visual_intersim.R``'s CSV output.

    Matrices use the *real* per-omic feature counts so the numpy side aligns
    feature-by-feature; values themselves are random (no InterSIM needed).
    """

    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    feat = _feat_counts(reference)
    for omic in OMIC_LAYERS:
        density = rng.normal(size=(N_REP_DENSITY, SUBSAMPLE))
        np.savetxt(directory / f"density_{omic}.csv", density, delimiter=",")
        matrix = rng.normal(size=(N_SAMPLE, feat[omic]))
        np.savetxt(directory / f"matrix_{omic}.csv", matrix, delimiter=",")
    cluster = (np.arange(N_SAMPLE) % N_CLUSTER) + 1
    (directory / "cluster_ids.csv").write_text(
        "cluster\n" + "\n".join(str(c) for c in cluster) + "\n"
    )
    (directory / "provenance.txt").write_text(
        "\n".join(
            [
                "intersim_version: test",
                "r_version: test",
                "generation_date: 2026-06-04",
                "base_seed: 123",
                f"n_sample: {N_SAMPLE}",
                f"n_cluster: {N_CLUSTER}",
                "delta: 2.0",
                "p_dmp: 0.2",
                f"n_rep_density: {N_REP_DENSITY}",
                f"subsample: {SUBSAMPLE}",
            ]
        )
        + "\n"
    )


@pytest.fixture(scope="module")
def packed_fixture(tmp_path_factory, reference):
    export = tmp_path_factory.mktemp("visual_export")
    _write_synthetic_export(export, reference)
    out = tmp_path_factory.mktemp("visual_fixture") / "fixture.npz"
    build_visual_fixture_from_export(export, out)
    return out


def test_load_visual_fixture_missing_raises(tmp_path):
    with pytest.raises(FidelityError, match="visual fixture not found"):
        load_visual_fixture(tmp_path / "absent.npz")


def test_build_and_load_roundtrip(packed_fixture, reference):
    fixture = load_visual_fixture(packed_fixture)
    feat = _feat_counts(reference)
    assert fixture.n_sample == N_SAMPLE
    assert fixture.n_cluster == N_CLUSTER
    for omic in OMIC_LAYERS:
        assert fixture.data.matrices[omic].shape == (N_SAMPLE, feat[omic])
        assert fixture.data.density[omic].shape == (N_REP_DENSITY, SUBSAMPLE)
    assert fixture.data.cluster_ids.shape == (N_SAMPLE,)
    assert set(fixture.data.cluster_ids).issubset(set(range(1, N_CLUSTER + 1)))


def test_generate_numpy_visual_data_matches_shapes(packed_fixture, reference):
    fixture = load_visual_fixture(packed_fixture)
    nd = generate_numpy_visual_data(fixture, reference=reference)
    for omic in OMIC_LAYERS:
        assert nd.matrices[omic].shape == fixture.data.matrices[omic].shape
        assert nd.density[omic].shape == fixture.data.density[omic].shape
    assert nd.cluster_ids.shape == (N_SAMPLE,)


def test_figure_builders_smoke(packed_fixture, reference):
    fixture = load_visual_fixture(packed_fixture)
    numpy_data = generate_numpy_visual_data(fixture, reference=reference)
    assert isinstance(numpy_data, VisualData)
    for name, builder in _FIGURE_BUILDERS.items():
        fig = builder(fixture.data, numpy_data)
        assert isinstance(fig, Figure), name
        plt.close(fig)


def test_run_fidelity_visual_writes_pngs(packed_fixture, tmp_path, reference):
    saved = run_fidelity_visual(packed_fixture, tmp_path, reference=reference)
    assert set(saved) == set(_FIGURE_BUILDERS)
    for path in saved.values():
        assert path.exists() and path.stat().st_size > 0
