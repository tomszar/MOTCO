from __future__ import annotations

import logging
import multiprocessing
import os
from typing import Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from motco.stats.trajectory import estimate_betas, estimate_difference

logger = logging.getLogger(__name__)


class _RRPPWorker:
    """Picklable callable that runs a chunk of RRPP permutations."""

    def __init__(
        self,
        y_hat: np.ndarray,
        y_res: np.ndarray,
        model_full: Union[pd.DataFrame, np.ndarray],
        ls_means: Union[pd.DataFrame, np.ndarray],
        contrast: list[list[int]],
    ) -> None:
        self.y_hat = y_hat
        self.y_res = y_res
        self.model_full = model_full
        self.ls_means = ls_means
        self.contrast = contrast

    def __call__(self, n_iters: int, seed: int):
        rng = np.random.default_rng(seed)
        n = self.y_res.shape[0]
        out_d, out_a, out_s = [], [], []
        for _ in range(n_iters):
            idx = rng.permutation(n)
            y_random = self.y_hat + self.y_res[idx, :]
            d, a, s = estimate_difference(y_random, self.model_full, self.ls_means, self.contrast)
            out_d.append(d)
            out_a.append(a)
            out_s.append(s)
        return out_d, out_a, out_s


def RRPP(
    Y: Union[pd.DataFrame, np.ndarray],
    model_full: Union[pd.DataFrame, np.ndarray],
    model_reduced: Union[pd.DataFrame, np.ndarray],
    LS_means: Union[pd.DataFrame, np.ndarray],
    contrast: list[list[int]],
    permutations: int = 999,
    n_jobs: Optional[int] = None,
    progress: bool = True,
) -> tuple:
    """
    Residual Randomization in a Permutation Procedure to evaluate
    linear models.

    Parameters
    ----------
    Y: Union[pd.DataFrame, np.ndarray]
        Outcome matrix.
    model_full: Union[pd.DataFrame, np.ndarray]
        Model matrix for full model, including intercept.
    model_reduced: Union[pd.DataFrame, np.ndarray]
        Model matrix for reduced model, including intercept.
    LS_means: Union[pd.DataFrame, np.ndarray]
        Least-squares means to estimate.
    contrast: list[list[int]]
        Indices indicating the groups to compare based on LS means.
        Each list must contain the cohorts that belong to the same group.
    permutations: int
        Number of permutations.
    n_jobs: Optional[int]
        If provided and > 1, run permutations in parallel using multiple
        worker processes. Use -1 to use all available CPUs. When None or 1,
        runs single-threaded (backward-compatible default).

    Returns
    -------
    dist_delta: list[float]
        Distribution of deltas.
    dist_angle: list[float]
        Distribution of angles.
    """
    # --- Input validation ---
    _Y = np.asarray(Y, dtype=float)
    _Xf = np.asarray(model_full, dtype=float)
    _Xr = np.asarray(model_reduced, dtype=float)
    _LS = np.asarray(LS_means, dtype=float)
    if _Y.shape[0] != _Xf.shape[0]:
        raise ValueError(
            f"Y has {_Y.shape[0]} rows but model_full has {_Xf.shape[0]} rows — "
            "number of rows must match."
        )
    if _Y.shape[0] != _Xr.shape[0]:
        raise ValueError(
            f"Y has {_Y.shape[0]} rows but model_reduced has {_Xr.shape[0]} rows — "
            "number of rows must match."
        )
    if _LS.shape[1] != _Xf.shape[1]:
        raise ValueError(
            f"LS_means has {_LS.shape[1]} columns but model_full has {_Xf.shape[1]} columns — "
            "number of columns must match."
        )
    _n_ls = _LS.shape[0]
    for _gi, _group in enumerate(contrast):
        for _idx in _group:
            if not (0 <= _idx < _n_ls):
                raise ValueError(
                    f"contrast[{_gi}] contains index {_idx}, but LS_means only has {_n_ls} rows "
                    f"(valid indices: 0–{_n_ls - 1})."
                )
    if not np.all(np.isfinite(_Y)):
        raise ValueError("Y contains NaN or Inf values.")
    if not np.all(np.isfinite(_Xf)):
        raise ValueError("model_full contains NaN or Inf values.")
    if not np.all(np.isfinite(_Xr)):
        raise ValueError("model_reduced contains NaN or Inf values.")
    # --- End validation ---
    # Set Y to be pandas df
    Y = pd.DataFrame(Y)
    # Set-up permutation procedure
    betas_red = estimate_betas(model_reduced, Y)
    # Predicted values from reduced model
    y_hat = np.matmul(model_reduced, betas_red)
    y_hat.index = Y.index
    # Resdiuals of reduced mode (these are the permuted units)
    y_res = Y - y_hat
    # Prepare NumPy views to avoid pandas inside the permutation loop
    y_hat_np = np.asarray(y_hat)
    y_res_np = np.asarray(y_res)
    # Prepare containers
    deltas: list[np.ndarray] = []
    angles: list[np.ndarray] = []
    shapes: list[np.ndarray] = []

    # Serial path
    if n_jobs in (None, 1):
        n = y_res_np.shape[0]
        rng = np.random.default_rng()
        for _ in tqdm(range(permutations), desc="RRPP", unit="perm", disable=not progress):
            idx = rng.permutation(n)
            y_random = y_hat_np + y_res_np[idx, :]
            d, a, s = estimate_difference(y_random, model_full, LS_means, contrast)
            deltas.append(d)
            angles.append(a)
            shapes.append(s)
        return deltas, angles, shapes

    # Parallel path
    n_workers = (os.cpu_count() or 1) if n_jobs == -1 else max(1, n_jobs or 1)
    n_workers = min(n_workers, max(1, permutations))
    logger.info("Running %d permutations across %d workers", permutations, n_workers)
    base = permutations // n_workers
    rem = permutations % n_workers
    counts = [base + (1 if i < rem else 0) for i in range(n_workers)]

    ss = np.random.SeedSequence()
    seeds = [int(s.generate_state(1)[0]) for s in ss.spawn(n_workers)]

    worker = _RRPPWorker(y_hat_np, y_res_np, model_full, LS_means, contrast)
    with multiprocessing.Pool(processes=n_workers) as pool:
        parts = pool.starmap(worker, zip(counts, seeds))

    for d_list, a_list, s_list in parts:
        deltas.extend(d_list)
        angles.extend(a_list)
        shapes.extend(s_list)

    return deltas, angles, shapes
