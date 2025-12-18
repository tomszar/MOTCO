from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import multiprocessing
import os

# Global context for RRPP multiprocessing workers
_RRPP_CTX: dict[str, object] = {}


def _rrpp_pool_init(
    y_hat_arr: np.ndarray,
    y_res_arr: np.ndarray,
    model_full_obj: Union[pd.DataFrame, np.ndarray],
    ls_means_obj: Union[pd.DataFrame, np.ndarray],
    contrast_obj: list[list[int]],
) -> None:
    _RRPP_CTX["y_hat"] = y_hat_arr
    _RRPP_CTX["y_res"] = y_res_arr
    _RRPP_CTX["model_full"] = model_full_obj
    _RRPP_CTX["ls_means"] = ls_means_obj
    _RRPP_CTX["contrast"] = contrast_obj


def _rrpp_pool_worker(n_iters: int, seed: int):
    rng = np.random.default_rng(seed)
    y_hat_arr = _RRPP_CTX["y_hat"]  # type: ignore[assignment]
    y_res_arr = _RRPP_CTX["y_res"]  # type: ignore[assignment]
    model_full_obj = _RRPP_CTX["model_full"]  # type: ignore[assignment]
    ls_means_obj = _RRPP_CTX["ls_means"]  # type: ignore[assignment]
    contrast_obj = _RRPP_CTX["contrast"]  # type: ignore[assignment]

    n = int(np.asarray(y_res_arr).shape[0])
    out_d, out_a, out_s = [], [], []
    for _ in range(n_iters):
        idx = rng.permutation(n)
        y_random = y_hat_arr + y_res_arr[idx, :]
        d, a, s = estimate_difference(y_random, model_full_obj, ls_means_obj, contrast_obj)
        out_d.append(d)
        out_a.append(a)
        out_s.append(s)
    return out_d, out_a, out_s


def center_matrix(
    dat: pd.DataFrame,
    group_col: str,
    level_col: str,
    feature_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Center feature columns by per-group means.

    Parameters
    ----------
    dat: pd.DataFrame
        Original, non-centered dataframe.
    group_col: str
        Column in `dat` indicating the group (between-subject factor).
    level_col: str
        Column in `dat` indicating the level/state (within-group factor).
    feature_cols: Sequence[str] | None
        Feature columns to center. If None, all numeric columns except
        `group_col` and `level_col` are used.

    Returns
    -------
    pd.DataFrame
        A copy of `dat` with selected feature columns centered within groups.
    """
    datc = dat.copy()
    if feature_cols is None:
        feature_cols = [
            c
            for c in datc.select_dtypes(include=[np.number]).columns.tolist()
            if c not in {group_col, level_col}
        ]
    if not feature_cols:
        return datc
    # Center within groups using group-wise means
    datc.loc[:, feature_cols] = (
        datc.loc[:, feature_cols]
        - datc.groupby(group_col)[feature_cols].transform("mean")
    )
    return datc


def get_model_matrix(
    X: pd.DataFrame,
    group_col: str,
    level_col: str,
    full: bool = True,
) -> np.ndarray:
    """
    Build a design (model) matrix for group × level factors.

    Coding scheme
    -------------
    - Intercept (column of ones).
    - Group main effects: one-hot with drop-first for groups (G-1 columns).
    - Level main effects: one-hot with drop-first for levels (L-1 columns).
    - If `full=True`, include all interaction terms between group and level
      dummies: (G-1) × (L-1) columns.

    The category order is deterministic: sorted by string representation.

    Parameters
    ----------
    X: pd.DataFrame
        DataFrame containing `group_col` and `level_col`.
    group_col: str
        Name of the group column.
    level_col: str
        Name of the level/state column.
    full: bool
        Whether to include interaction terms.

    Returns
    -------
    np.ndarray
        Model matrix with intercept.
    """
    # Determine deterministic category order
    g_levels = sorted(pd.unique(X[group_col].astype(str)).tolist())
    l_levels = sorted(pd.unique(X[level_col].astype(str)).tolist())
    g = pd.Categorical(X[group_col].astype(str), categories=g_levels, ordered=True)
    l = pd.Categorical(X[level_col].astype(str), categories=l_levels, ordered=True)

    G = pd.get_dummies(g, drop_first=True, dtype=int)
    L = pd.get_dummies(l, drop_first=True, dtype=int)

    parts = []
    # Intercept
    parts.append(pd.DataFrame({"Intercept": np.ones(len(X))}, index=X.index))
    # Main effects
    if G.shape[1] > 0:
        parts.append(G)
    if L.shape[1] > 0:
        parts.append(L)
    # Interactions
    if full and G.shape[1] > 0 and L.shape[1] > 0:
        inter_cols = {}
        for g_col in G.columns:
            for l_col in L.columns:
                inter_cols[f"{g_col}:{l_col}"] = G[g_col].values * L[l_col].values
        parts.append(pd.DataFrame(inter_cols, index=X.index))

    model_mat = pd.concat(parts, axis=1).to_numpy()
    return model_mat


def pair_difference(
    dat: pd.DataFrame,
    group_col: str,
    level_col: str,
    groups: tuple[str, str] | None = None,
    levels: tuple[str, str] | None = None,
    feature_cols: Sequence[str] | None = None,
) -> tuple[float, float]:
    """
    Estimate difference in direction (angle, degrees) and magnitude (delta)
    between two groups across two levels.

    The change vector for a group is defined as `level1 - level2` over the
    selected feature columns.

    Parameters
    ----------
    dat: pd.DataFrame
        DataFrame containing features plus `group_col` and `level_col`.
    group_col: str
        Column with groups (between-subject factor).
    level_col: str
        Column with levels/states (within-subject factor).
    groups: tuple[str, str] | None
        Pair of group labels to compare. If None, infer and require exactly two.
    levels: tuple[str, str] | None
        Pair of level labels to use for the change vector. If None, infer and
        require exactly two.
    feature_cols: Sequence[str] | None
        Feature columns to use. If None, all numeric columns except `group_col`
        and `level_col` are used.

    Returns
    -------
    tuple[float, float]
        (angle_degrees, delta_magnitude_difference)

    Notes
    -----
    See [1]_ for more information on two-state comparisons.

    References
    ----------
    .. [1] Collyer, Michael L., and Dean C. Adams. "Analysis of two‐state
           multivariate phenotypic change in ecological studies." Ecology 88.3
           (2007): 683-692. https://doi.org/10.1890/06-0727
    """
    if feature_cols is None:
        feature_cols = [
            c
            for c in dat.select_dtypes(include=[np.number]).columns.tolist()
            if c not in {group_col, level_col}
        ]
    if not feature_cols:
        raise ValueError("No feature columns provided or detected.")

    g_vals = sorted(pd.unique(dat[group_col].astype(str)).tolist())
    l_vals = sorted(pd.unique(dat[level_col].astype(str)).tolist())
    if groups is None:
        if len(g_vals) != 2:
            raise ValueError(
                f"Expected exactly 2 groups, found {len(g_vals)}: {g_vals}"
            )
        groups = (g_vals[0], g_vals[1])
    if levels is None:
        if len(l_vals) != 2:
            raise ValueError(
                f"Expected exactly 2 levels, found {len(l_vals)}: {l_vals}"
            )
        levels = (l_vals[0], l_vals[1])

    # Compute per (group, level) means
    means = (
        dat.assign(
            __g=dat[group_col].astype(str), __l=dat[level_col].astype(str)
        )
        .groupby(["__g", "__l"])[feature_cols]
        .mean()
    )
    try:
        y_g1 = means.loc[(groups[0], levels[0])] - means.loc[(groups[0], levels[1])]
        y_g2 = means.loc[(groups[1], levels[0])] - means.loc[(groups[1], levels[1])]
    except KeyError as e:
        raise ValueError(
            "Missing combinations for the requested groups/levels in the data."
        ) from e

    d1 = float(np.linalg.norm(y_g1.values))
    d2 = float(np.linalg.norm(y_g2.values))
    delta = abs(d1 - d2)
    if d1 == 0 or d2 == 0:
        raise ValueError(
            "Zero-magnitude change vector for at least one group; angle is undefined."
        )
    angle = float(np.degrees(np.arccos(np.inner(y_g1 / d1, y_g2 / d2))))
    return angle, delta


def estimate_difference(
    Y: Union[pd.DataFrame, np.ndarray],
    model_matrix: Union[pd.DataFrame, np.ndarray],
    LS_means: Union[pd.DataFrame, np.ndarray],
    contrast: list[list[int]],
) -> tuple:
    """
    Estimate parameters angle, delta, and shape given an outcome
    matrix, model matrix, and contrast to compare. This is a comparison
    of more than two states.

    Parameters
    ----------
    Y: Union[pd.DataFrame, np.ndarray]
        Outcome matrix.
    model_matrix: Union[pd.DataFrame, np.ndarray]
        Model matrix with intercept.
    LS_means: Union[pd.DataFrame, np.ndarray]
        Least-squares means to estimate.
    contrast: list[list[int]]
        Indices indicating the groups to compare based on LS means.
        Each list must contain the cohorts that belong to the same group.

    Returns
    -------
    deltas: np.ndarray
        Symmetric matrix (n_groups x n_groups) with differences in magnitude.
    angles: np.ndarray
        Symmetric matrix (n_groups x n_groups) with differences in direction (degrees).
    shapes: np.ndarray
        Symmetric matrix (n_groups x n_groups) with shape distances.

    Notes
    -----
    See [1]_ for more information on trajectory analysis.

    References
    ----------
    .. [1] Adams, Dean C., and Michael L. Collyer.
           "A general framework for the analysis of phenotypic trajectories in
           evolutionary studies."
           Evolution: International Journal of Organic Evolution 63.5 (2009):
           1143-1154.
           https://doi.org/10.1111/j.1558-5646.2009.00649.x
    """
    n_groups = len(contrast)
    betas = estimate_betas(model_matrix, Y)
    # Compute LS-mean vectors; keep as DataFrame to minimize behavioral drift
    obs_vect = pd.DataFrame(
        np.matmul(np.asarray(LS_means, dtype=float), np.asarray(betas, dtype=float))
    )
    ys = []
    des = []
    angles = np.zeros((n_groups, n_groups))
    deltas = np.zeros((n_groups, n_groups))
    for i in range(n_groups):
        y = _estimate_orientation(obs_vect, contrast[i])
        d = _estimate_size(obs_vect, contrast[i])
        des.append(d)
        ys.append(y)
    shapes = _estimate_shape(obs_vect, contrast)
    for i in range(n_groups):
        comp = i + 1
        while comp < n_groups:
            delta = np.abs(des[i] - des[comp])
            # When using SVD, no need to divide by size
            angle = np.arccos(np.inner(ys[i], ys[comp])) * 180 / np.pi
            deltas[i, comp] = delta
            deltas[comp, i] = delta
            angles[i, comp] = angle
            angles[comp, i] = angle
            comp += 1
    return deltas, angles, shapes


def RRPP(
    Y: Union[pd.DataFrame, np.ndarray],
    model_full: Union[pd.DataFrame, np.ndarray],
    model_reduced: Union[pd.DataFrame, np.ndarray],
    LS_means: Union[pd.DataFrame, np.ndarray],
    contrast: list[list[int]],
    permutations: int = 999,
    n_jobs: Optional[int] = None,
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

    # If no parallelization requested, keep original serial behavior
    if n_jobs in (None, 1):
        n = int(np.asarray(y_res_np).shape[0])
        rng = np.random.default_rng()
        for _ in range(permutations):
            idx = rng.permutation(n)
            # Create randomized response
            y_random = y_hat_np + y_res_np[idx, :]
            d, a, s = estimate_difference(y_random, model_full, LS_means, contrast)
            deltas.append(d)
            angles.append(a)
            shapes.append(s)
        return deltas, angles, shapes

    # Parallel path (process-based): perform permutations with NumPy only inside workers
    # to reduce pandas overhead and IPC volume.
    y_hat_np = y_hat.to_numpy()
    y_res_np = y_res.to_numpy()

    # Determine number of workers and distribute work
    if n_jobs == -1 or n_jobs is None:
        n_workers = os.cpu_count() or 1
    else:
        n_workers = max(1, int(n_jobs))
    n_workers = min(n_workers, max(1, permutations))
    base = permutations // n_workers
    rem = permutations % n_workers
    counts = [base + (1 if i < rem else 0) for i in range(n_workers)]

    # Create independent seeds per worker
    ss = np.random.SeedSequence()
    seeds = [int(s.generate_state(1)[0]) for s in ss.spawn(n_workers)]

    with multiprocessing.Pool(
        processes=n_workers,
        initializer=_rrpp_pool_init,
        initargs=(y_hat_np, y_res_np, model_full, LS_means, contrast),
    ) as pool:
        parts = pool.starmap(_rrpp_pool_worker, zip(counts, seeds))

    # Concatenate results preserving list-of-matrices shape
    for d_list, a_list, s_list in parts:
        deltas.extend(d_list)
        angles.extend(a_list)
        shapes.extend(s_list)

    return deltas, angles, shapes


def estimate_betas(
    X: Union[pd.DataFrame, np.ndarray], Y: Union[pd.DataFrame, np.ndarray]
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Estimate the beta coefficients between an outcome matrix
    and a model matrix

    Parameters
    ----------
    X: Union[pd.DataFrame, np.ndarray]
        Model matrix with intercept.
    Y: Union[pd.DataFrame, np.ndarray]
        Outcome matrix.

    Returns
    -------
    betas: Union[pd.DataFrame, np.ndarray]
        Beta coefficients
    """
    # Convert inputs to arrays for linear algebra while preserving Y's metadata
    X_arr = np.asarray(X, dtype=float)
    Y_is_df = isinstance(Y, pd.DataFrame)
    if Y_is_df:
        Y_arr = Y.to_numpy(dtype=float)
        y_cols = Y.columns
    else:
        Y_arr = np.asarray(Y, dtype=float)
        y_cols = None

    # Solve normal equations using factorization with robust fallbacks
    XtX = X_arr.T @ X_arr
    XtY = X_arr.T @ Y_arr
    try:
        # Cholesky is fastest and most stable for SPD XtX
        L = np.linalg.cholesky(XtX)
        tmp = np.linalg.solve(L, XtY)
        betas_arr = np.linalg.solve(L.T, tmp)
    except np.linalg.LinAlgError:
        try:
            # Fall back to a direct solve of the normal equations
            betas_arr = np.linalg.solve(XtX, XtY)
        except np.linalg.LinAlgError:
            # Final fallback: least-squares without forming normal equations
            # This handles rank deficiency and ill-conditioning better.
            betas_arr, *_ = np.linalg.lstsq(X_arr, Y_arr, rcond=None)

    if Y_is_df:
        # Return a DataFrame so downstream matmul with numpy yields a DataFrame
        # and index/column handling stays consistent with previous behavior.
        return pd.DataFrame(betas_arr, columns=y_cols)
    return betas_arr


def get_observed_vectors(
    X: pd.DataFrame,
    Y: Union[pd.DataFrame, np.ndarray],
    group_col: str,
    level_col: str,
    full: bool = True,
) -> pd.DataFrame:
    """
    Get LS-mean vectors for each group × level cell.

    Parameters
    ----------
    X: pd.DataFrame
        DataFrame containing factors `group_col` and `level_col` for each row
        corresponding to `Y`.
    Y: Union[pd.DataFrame, np.ndarray]
        Outcome matrix (n_samples × n_features).
    group_col: str
        Group column name in `X`.
    level_col: str
        Level/state column name in `X`.
    full: bool
        Whether to include interactions in the model.

    Returns
    -------
    pd.DataFrame
        LS means arranged with a MultiIndex (group, level). Columns follow `Y`.
    """
    model_full = get_model_matrix(X[[group_col, level_col]], group_col, level_col, full)
    betas = estimate_betas(model_full, Y)

    g_levels = sorted(pd.unique(X[group_col].astype(str)).tolist())
    l_levels = sorted(pd.unique(X[level_col].astype(str)).tolist())
    ls_matrix = build_ls_means(g_levels, l_levels, full)
    means = np.matmul(ls_matrix, betas)

    # Build a clear index and columns
    idx = pd.MultiIndex.from_product([g_levels, l_levels], names=[group_col, level_col])
    if isinstance(Y, pd.DataFrame):
        cols = Y.columns
    else:
        cols = [f"f{i}" for i in range(means.shape[1])]
    return pd.DataFrame(means, index=idx, columns=cols)


def _estimate_size(obs_vect: pd.DataFrame | np.ndarray, levels: list[int]) -> float:
    """
    Estimate the size of a trajectory of two or more levels.

    Parameters
    ----------
    obs_vect: pd.DataFrame
        Matrix of observed mean vectors.
    levels: list[int]
        List of indices indicating the levels to consider.

    Returns
    -------
    size: float
        Size of the trajectory.
    """
    # Use a fully vectorized NumPy implementation to avoid Python-loop overhead
    X = np.asarray(obs_vect, dtype=float)[levels, :]
    if X.shape[0] < 2:
        return 0.0
    diffs = X[:-1] - X[1:]
    size = np.linalg.norm(diffs, axis=1).sum()
    return float(size)


def _estimate_orientation(
    obs_vect: pd.DataFrame | np.ndarray,
    levels: list[int],
) -> np.ndarray:
    """
    Estimate the orientation of a trajectory of two or more levels.

    Parameters
    ----------
    obs_vect: pd.DataFrame
        Matrix of observed mean vectors.
    levels: list[int]
        List of indices indicating the levels to consider.

    Returns
    -------
    orientation: int
        Orientation of the trajectory.
    """
    # Vectorized implementation using symmetric-eigen decomposition on covariance
    X = np.asarray(obs_vect, dtype=float)[levels, :]
    # Center rows
    X = X - X.mean(axis=0, keepdims=True)
    # Sample covariance (symmetric); eigh is efficient for symmetric matrices
    n = X.shape[0]
    if n > 1:
        C = (X.T @ X) / (n - 1)
    else:
        # Degenerate case: fall back to outer product (all zeros if single row)
        C = X.T @ X
    # Eigenvalues ascending; last eigenvector corresponds to principal component
    w, v = np.linalg.eigh(C)
    orientation = v[:, -1]
    # Ensure deterministic sign following the first (centered) vector
    c1 = float(orientation @ X[0, :])
    if c1 < 0:
        orientation = -orientation
    return orientation


def _estimate_shape(
    vectors: Union[pd.DataFrame, np.ndarray], contrast: list[list[int]]
) -> np.ndarray:
    """
    Align shapes using procrustes superimpostion and estimate shape
    differences.

    Parameters
    ----------
    vectors: Union[pd.DataFrame, np.ndarray]
        A n by k point matrix with the vectors to align,
        where n is the number of points, and k the number of dimensions.
    contrast: list[list[int]]
        Indices indicating the groups to compare based on LS means.
        Each list within the list must contain the cohorts that belong
        to the same group.

    Returns
    -------
    shape_distance: np.ndarray
        Matrix with shape distances.
    """
    # Implement R-equivalent GPA (pgpa + pPsup): similarity Procrustes with per-iteration
    # re-centering and re-scaling, and beta scaling = sum of (possibly signed) singular values.
    V = np.asarray(vectors, dtype=float)
    n_groups = len(contrast)
    n_levels = len(contrast[0])
    n_dimensions = V.shape[1]

    # Build (G, L, K) tensor of vectors per group and level once
    X = np.empty((n_groups, n_levels, n_dimensions), dtype=float)
    for gi, levels in enumerate(contrast):
        X[gi] = V[np.asarray(levels, dtype=int), :]

    # Helper: center then scale by centroid size (Frobenius norm of centered matrix)
    def _center_scale_unit(A: np.ndarray) -> np.ndarray:
        Z = A - A.mean(axis=0, keepdims=True)
        cs = float(np.linalg.norm(Z))
        if not np.isfinite(cs) or cs <= 1e-15:
            cs = 1.0
        return Z / cs

    # Initialize temp1 as in R: temp1[,,i] <- trans(csize(A[,,i])[[2]])
    temp1 = np.empty_like(X)
    for i in range(n_groups):
        temp1[i] = _center_scale_unit(X[i])

    # Distance matrix of flattened shapes (like dist(t(matrix(...))))
    def _pairwise_flat_dist(arr: np.ndarray) -> np.ndarray:
        flat = arr.reshape((n_groups, n_dimensions * n_levels))
        return euclidean_distances(flat)

    Qm1 = _pairwise_flat_dist(temp1)
    # Sum of lower triangle (no diagonal) as in R's sum(dist(.))
    Q_prev_sum = float(np.tril(Qm1, k=-1).sum())
    Q_improve = Q_prev_sum  # initialize to enter loop

    # Iterate until improvement is negligible
    while abs(Q_improve) > 0.00001:
        temp2 = np.empty_like(temp1)
        for i in range(n_groups):
            # Mean shape of all groups except i (mshape(temp1[,,-i]))
            if n_groups > 1:
                M = temp1[np.arange(n_groups) != i].mean(axis=0)
            else:
                M = temp1[i]

            # pPsup equivalent: re-center and re-scale both shapes
            Z1 = _center_scale_unit(temp1[i])
            Z2 = _center_scale_unit(M)

            # Cross-covariance and SVD
            H = Z2.T @ Z1
            U, S, Vt = np.linalg.svd(H, full_matrices=False)
            V = Vt.T

            # R mapping to R code: U_R <- V_np; V_R <- U_np
            # sig <- sign(det(t(Z1) %*% Z2)) == sign(det(H))
            detH = float(np.linalg.det(H))
            sig = -1.0 if detH < 0.0 else 1.0  # treat 0 as +1

            # Flip last column of V_R (which is U here) by sig
            U[:, -1] *= sig

            # Gam <- U_R %*% t(V_R) == V @ U.T
            Gam = V @ U.T

            # beta <- sum(Delt) with last singular value signed by sig
            if S.size == 0:
                beta = 0.0
            elif S.size == 1:
                beta = float(sig * S[0])
            else:
                beta = float(np.sum(S[:-1]) + sig * S[-1])

            # Mp1 = beta * Z1 %*% Gam
            temp2[i] = beta * (Z1 @ Gam)

        # Convergence check
        Qm2 = _pairwise_flat_dist(temp2)
        Q_sum = float(np.tril(Qm2, k=-1).sum())
        Q_improve = Q_prev_sum - Q_sum
        Q_prev_sum = Q_sum
        temp1 = temp2

    return Qm2


def _OPA(M1: np.ndarray, M2: np.ndarray) -> np.ndarray:
    """
    Given two matrices, rotate M2 to perfectly align with M1
    using Orthogonal Procrustes Analysis [1]_.

    Parameters
    ----------
    M1: np.ndarray
        Reference matrix to use.
    M2: np.ndarray
        Target matrix to change.

    Returns
    -------
    Mp2: np.ndarray
        Target matrix rotated.

    References
    ----------
    .. [1] Rohlf, F. James, and Dennis Slice.
           "Extensions of the Procrustes method for the optimal superimposition
           of landmarks." Systematic biology 39.1 (1990): 40-59.
           https://doi.org/10.2307/2992207
    """
    # Minimal Kabsch implementation with reflection correction
    # Compute covariance
    H = M1.T @ M2
    # SVD of covariance
    U, _, Vt = np.linalg.svd(H, full_matrices=False)
    # Rotation
    R = Vt.T @ U.T
    # Ensure a proper rotation (no reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    # Rotate M2
    Mp2 = M2 @ R
    return Mp2


def build_ls_means(
    group_levels: Sequence[str],
    level_levels: Sequence[str],
    full: bool = True,
) -> np.ndarray:
    """
    Generate LS-mean rows for every group × level cell consistent with
    `get_model_matrix` coding.

    Parameters
    ----------
    group_levels: Sequence[str]
        Sorted group labels; first is baseline.
    level_levels: Sequence[str]
        Sorted level labels; first is baseline.
    full: bool
        Whether to include interaction terms.

    Returns
    -------
    np.ndarray
        LS-mean design matrix with shape (G×L, 1 + (G-1) + (L-1) + I), where
        I = (G-1)×(L-1) if `full=True` else 0. Row order is by group major,
        then level minor.
    """
    g_levels = list(group_levels)
    l_levels = list(level_levels)
    Gm1 = max(len(g_levels) - 1, 0)
    Lm1 = max(len(l_levels) - 1, 0)
    n_rows = max(len(g_levels), 1) * max(len(l_levels), 1)
    n_cols = 1 + Gm1 + Lm1 + (Gm1 * Lm1 if full else 0)
    M = np.zeros((n_rows, n_cols), dtype=float)

    def row_for(i_g: int, i_l: int) -> int:
        return i_g * len(l_levels) + i_l

    # Column indices
    col = 0
    INTERCEPT = col
    col += 1
    G_START = col
    col += Gm1
    L_START = col
    col += Lm1
    I_START = col if full else None

    for gi, g_val in enumerate(g_levels):
        for li, l_val in enumerate(l_levels):
            r = row_for(gi, li)
            # Intercept
            M[r, INTERCEPT] = 1.0
            # Group dummies (drop first)
            if gi > 0 and Gm1 > 0:
                M[r, G_START + (gi - 1)] = 1.0
            # Level dummies (drop first)
            if li > 0 and Lm1 > 0:
                M[r, L_START + (li - 1)] = 1.0
            # Interactions
            if full and gi > 0 and li > 0 and (Gm1 > 0 and Lm1 > 0):
                idx = (gi - 1) * Lm1 + (li - 1)
                M[r, I_START + idx] = 1.0
    return M
