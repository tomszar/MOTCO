"""Similarity Network Fusion and spectral embedding utilities.

This module implements:
- `get_affinity_matrix`: builds per-dataset affinity matrices from squared
  Euclidean distances using an RBF-like kernel with local scaling.
- `SNF`: cross-diffusion across multiple networks to obtain a fused similarity.
- `get_spectral`: spectral embedding from a fused (or any) affinity matrix.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.manifold import spectral_embedding


def SNF(Ws: list[np.ndarray], k: int = 20, t: int = 20) -> np.ndarray:
    """
    Similarity Network Fusion (SNF) across multiple affinity matrices.

    The algorithm iteratively performs cross-diffusion using k-nearest
    neighbor sparse kernels and averages information from the other
    networks, producing a fused similarity matrix.

    Parameters
    ----------
    Ws: list[np.ndarray]
        List of affinity matrices to fuse. All matrices must have the same
        shape (n_samples x n_samples) and be symmetric.
    k: int
        Number of nearest neighbors for the sparse kernels. Default 20.
    t: int
        Number of cross-diffusion iterations. Default 20.

    Returns
    -------
    Pc: np.ndarray
        Fused similarity matrix (n_samples x n_samples).
    """
    nw = len(Ws)
    if nw < 2:
        raise ValueError("SNF requires at least two affinity matrices")

    Ps: list[np.ndarray] = []
    Ss: list[np.ndarray] = []

    for i in range(nw):
        Pi = _full_kernel(Ws[i])
        Pi = (Pi + Pi.T) / 2
        Ps.append(Pi)
        Ss.append(_sparse_kernel(Pi, k))

    # Initialize states
    Pst0 = [p.copy() for p in Ps]
    Pst1 = [p.copy() for p in Ps]

    # Iterations
    for _ in range(t):
        for j in range(nw):
            # Average of all other networks
            others = [Pst0[m] for m in range(nw) if m != j]
            M = sum(others) / (nw - 1)
            Pst1[j] = Ss[j] @ M @ Ss[j].T
            Pst1[j] = _full_kernel(Pst1[j])
        Pst0 = [p.copy() for p in Pst1]

    Pc = sum(Pst1) / nw
    return Pc


def get_affinity_matrix(
    dats: list[np.ndarray], K: int = 20, eps: float = 0.5
) -> list[np.ndarray]:
    """
    Estimate the affinity matrix for all datasets in dats from the squared Euclidean
    distance.

    Parameters
    ----------
    dats: list[np.ndarray]
        list of data sets to estimate the affinity matrix.
    K: int
        Number of K nearest neighbors to use. Default 20.
    eps: float
        Normalization factor. Recommended between 0.3 and 0.8. Default 0.5.

    Returns
    -------
    Ws: list[np.ndarray]
        list of affinity matrices
    """
    nrows = len(dats[0])
    Ws: list[np.ndarray] = []
    for dat in dats:
        arr = np.asarray(dat)
        euc_dist = cdist(arr, arr, metric="euclidean") ** 2
        Ws.append(_affinity_matrix(euc_dist, K, eps))

    return Ws


def get_spectral(aff: np.ndarray) -> np.ndarray:
    """
    Calculate spectral embedding from an affinity/similarity matrix.

    Parameters
    ----------
    aff: np.ndarray
        Affinity matrix to calculate the spectral embedding.

    Returns
    -------
    embedding: np.ndarray
        Spectral embedding.
    """
    embedding = spectral_embedding(aff, n_components=10, random_state=1548)
    return embedding


def _full_kernel(W: np.ndarray) -> np.ndarray:
    """
    Calculate full kernel matrix normalization.

    Parameters
    ----------
    W: np.ndarray
        Matrix in which to perform the full kernel normalization.

    Returns
    -------
    P: np.ndarray
        Normalized matrix.
    """

    rowsum = W.sum(axis=1) - W.diagonal()
    rowsum[rowsum == 0] = 1
    P = W / (2 * rowsum)

    np.fill_diagonal(P, 0.5)
    return P


def _sparse_kernel(W: np.ndarray, k: int) -> np.ndarray:
    """
    Calculate sparse kernel matrix using the k nearest neighbors.

    Parameters
    ----------
    W: np.ndarray
        Matrix in which to perform the sparse kernel matrix.
    k: int
        Number of nearest neighbors to use.

    Returns
    -------
    S: np.ndarray
        Sparse kernel matrix.
    """
    nrow = len(W)
    S = np.zeros((nrow, nrow))
    for i in range(0, nrow):
        s1 = W[i, :].copy()
        ix = s1.argsort()
        last = nrow - k
        s1[ix[0:last]] = 0
        S[i,] = s1

    S = _full_kernel(S)
    return S


def _euclidean_dist(dat: pd.DataFrame) -> np.ndarray:
    """
    Calculate the pairwise Euclidean distance between
    all the rows of a dataframe
    """
    euc_dist = cdist(dat, dat, metric="euclidean")
    euc_dist.index = dat.index
    euc_dist.columns = dat.index

    return euc_dist


def _affinity_matrix(mat, K, eps):
    Machine_Epsilon = np.finfo(float).eps
    Diff_mat = (mat + mat.transpose()) / 2
    # Sort distance matrix ascending order
    # (i.e. more similar is closer to first column)
    Diff_mat_sort = Diff_mat - np.diag(np.diag(Diff_mat))
    Diff_mat_sort = np.sort(Diff_mat_sort, axis=1)
    # Average distance with K nearest neighbors
    K_dist = np.mean(Diff_mat_sort[:, 1 : (K + 1)], axis=1) + Machine_Epsilon
    sigma = ((np.add.outer(K_dist, K_dist) + Diff_mat) / 3) + Machine_Epsilon
    sigma[sigma < Machine_Epsilon] = Machine_Epsilon

    W = stats.norm.pdf(Diff_mat, loc=0, scale=(eps * sigma))

    return (W + W.transpose()) / 2
