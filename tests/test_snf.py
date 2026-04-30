# tests/test_snf.py
from __future__ import annotations

import numpy as np
import pytest

from motco.stats.snf import get_affinity_matrix, SNF, get_spectral


def _datasets(n: int = 20, p: int = 5, n_datasets: int = 2, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.standard_normal((n, p)) for _ in range(n_datasets)]


def test_affinity_matrix_returns_correct_count():
    dats = _datasets()
    Ws = get_affinity_matrix(dats, K=5)
    assert len(Ws) == len(dats)


def test_affinity_matrix_shapes_are_square():
    dats = _datasets(n=20)
    Ws = get_affinity_matrix(dats, K=5)
    for W in Ws:
        assert W.shape == (20, 20)


def test_affinity_matrix_non_negative():
    dats = _datasets()
    Ws = get_affinity_matrix(dats, K=5)
    for W in Ws:
        assert np.all(W >= 0)


def test_affinity_matrix_symmetric():
    dats = _datasets()
    Ws = get_affinity_matrix(dats, K=5)
    for W in Ws:
        assert np.allclose(W, W.T)


def test_snf_output_shape():
    n = 20
    dats = _datasets(n=n)
    Ws = get_affinity_matrix(dats, K=5)
    fused = SNF(Ws, k=5, t=5)
    assert fused.shape == (n, n)


def test_snf_output_approximate_symmetric():
    dats = _datasets()
    Ws = get_affinity_matrix(dats, K=5)
    fused = SNF(Ws, k=5, t=5)
    # SNF output is approximately symmetric due to averaging steps
    assert np.allclose(fused, fused.T, atol=0.05)


def test_get_spectral_default_shape():
    dats = _datasets(n=20)
    Ws = get_affinity_matrix(dats, K=5)
    fused = SNF(Ws, k=5, t=5)
    emb = get_spectral(fused)
    assert emb.shape == (20, 10)  # default 10 components
