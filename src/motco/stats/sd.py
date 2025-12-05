from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import OneHotEncoder


def center_matrix(dat: pd.DataFrame) -> pd.DataFrame:
    """
    Center matrix grouped by sex.

    Parameters
    ----------
    dat: pd.DataFrame
        Original, non-centered dataframe

    Returns
    ----------
    datc: pd.DataFrame
        Centered dataframe
    """
    datc = dat.copy()
    dat_num = dat.copy()
    del dat_num["DX"]
    means = dat_num.groupby("PTGENDER").mean()
    last_col = str(dat.columns.get_loc("DX") - 1)
    datc.loc[dat["PTGENDER"] == "Female", "0":last_col] = (
        datc.loc[dat["PTGENDER"] == "Female", "0":last_col] - means.iloc[0, :]
    )
    datc.loc[dat["PTGENDER"] == "Male", "0":last_col] = (
        datc.loc[dat["PTGENDER"] == "Male", "0":last_col] - means.iloc[1, :]
    )
    return datc


def get_model_matrix(X: pd.DataFrame, full: bool = True) -> np.ndarray:
    """
    Generate a model matrix from a dataframe of factors.

    Parameters
    ----------
    X: pd.DataFrame
        Matrix with factors to convert to model matrix.
    full: bool
        Whether to return the full matrix. If False, then
        return the reduced matrix.

    Returns
    -------
    model_mat: np.ndarray
        Model matrix with intercept.
    """
    enc = OneHotEncoder(handle_unknown="ignore")
    model_mat = enc.fit(X).transform(X).toarray()
    # Intercept is control female
    model_mat = np.delete(model_mat, [2, 3], 1)
    if full:
        model_mat = np.c_[
            model_mat,
            model_mat[:, 0] * model_mat[:, 2],
            model_mat[:, 1] * model_mat[:, 2],
        ]
    # Add intercept
    model_mat = np.c_[np.ones(model_mat.shape[0]), model_mat]
    return model_mat


def pair_difference(dat: pd.DataFrame) -> tuple:
    """
    Estimate the difference in magnitude and direction
    in two states, divided by sex.

    Parameters
    ----------
    dat: pd.DataFrame
        Dataframe to estimate the magnitude, must have DX and PTGENDER
        columns.

    Returns
    ----------
    angle, delta: tuple
        Difference in direction, angle, and difference in magnitue, delta.

    Notes
    -----
    See [1]_ for more information on two-state comparisons.

    References
    ----------
    .. [1] Collyer, Michael L., and Dean C. Adams.
           "Analysis of two‐state multivariate phenotypic change in ecological
           studies." Ecology 88.3 (2007): 683-692.
           https://doi.org/10.1890/06-0727
    """
    last_col = dat.columns.get_loc("DX")
    means = {
        ("Female", "NL"): pd.DataFrame(),
        ("Female", "Dementia"): pd.DataFrame(),
        ("Male", "NL"): pd.DataFrame(),
        ("Male", "Dementia"): pd.DataFrame(),
    }
    for g, d in dat.groupby(["PTGENDER", "DX"]):
        if g in means:
            means[g] = d.iloc[:, :last_col].mean()
    yf = means[("Female", "NL")] - means[("Female", "Dementia")]
    ym = means[("Male", "NL")] - means[("Male", "Dementia")]
    Def = np.sqrt(np.sum(np.power(yf, 2)))
    Dem = np.sqrt(np.sum(np.power(ym, 2)))
    delta = Def - Dem
    angle = np.arccos(np.inner(yf / Def, ym / Dem)) * 180 / np.pi
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
    obs_vect = pd.DataFrame(np.matmul(LS_means, betas))
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
    ids = y_res.index
    deltas = []
    angles = []
    shapes = []
    for i in range(permutations):
        # Permute rows
        ids_permuted = np.random.permutation(ids)
        y_res_permuted = y_res.loc[ids_permuted, :]
        y_res_permuted.index = y_res.index
        # Create random values
        y_random = y_hat + y_res_permuted
        d, a, s = estimate_difference(y_random, model_full, LS_means, contrast)
        deltas.append(d)
        angles.append(a)
        shapes.append(s)
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
    left = np.matmul(np.transpose(X), X)
    right = np.matmul(np.transpose(X), Y)
    betas = np.matmul(np.linalg.inv(left), right)
    return betas


def get_observed_vectors(X, Y) -> pd.DataFrame:
    """
    Get means, or observed vectors, from standard LS vectors.

    Parameters
    ----------
    X: pd.DataFrame
        X matrix of responses.
    Y: pd.DataFrame
        Y matrix of outcomes.

    Returns
    -------
    means: pd.DataFrame
        Mean values.
    """
    model_full = get_model_matrix(X)
    betas = estimate_betas(model_full, Y)
    ls_matrix = _get_ls_vectors()
    means = np.matmul(ls_matrix, betas)

    return means


def _estimate_size(obs_vect: pd.DataFrame, levels: list[int]) -> int:
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
    size: int
        Size of the trajectory.
    """
    if isinstance(obs_vect, pd.DataFrame) is False:
        obs_vect = pd.DataFrame(obs_vect)
    n_levels = len(levels)
    size = 0
    for i, val in enumerate(levels):
        if val < levels[n_levels - 1]:
            y = obs_vect.iloc[val, :] - obs_vect.iloc[levels[i + 1], :]
            d = np.sqrt(np.sum(np.power(y, 2)))
            size += d
    return size


def _estimate_orientation(
    obs_vect: pd.DataFrame,
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
    if isinstance(obs_vect, pd.DataFrame) is False:
        obs_vect = pd.DataFrame(obs_vect)
    # N of dimensions
    vect = obs_vect.iloc[levels, :]
    k = vect.shape[1]
    # Center matrix
    vect -= np.mean(vect, axis=0)
    # SVD
    _, _, V = np.linalg.svd(np.cov(vect.transpose()))
    orientation = V.transpose()[:k, 0]
    # Check sing
    c1 = np.matmul(orientation, vect.iloc[0, :])
    sign = c1 / np.abs(c1)
    if sign < 0:
        orientation = -1 * orientation
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
    vect_c = np.array(vectors.copy())
    n_groups = len(contrast)
    n_levels = len(contrast[0])
    n_dimensions = vect_c.shape[1]
    for i, levels in enumerate(contrast):
        means = vectors.iloc[levels, :].mean()
        vect_c[levels, :] = vectors.iloc[levels, :] - means
    # Scale to centroid size
    for i, levels in enumerate(contrast):
        centroid = np.mean(vect_c[levels, :], axis=0)
        cs = np.sqrt(np.sum(np.sum(np.power(vect_c[levels, :] - centroid, 2))))
        vect_c[levels, :] = vect_c[levels, :] / cs
    # Get baseline Euclidean distance
    Qm1 = euclidean_distances(vect_c.reshape((n_groups, n_dimensions * n_levels)))
    Q = np.tril(Qm1).sum()
    temp1 = vect_c.copy()
    temp2 = vect_c.copy()
    iter = 0
    while abs(Q) > 0.00001:
        # Each shape against the mean of the rest
        for i, levels in enumerate(contrast):
            b = [x for ind, x in enumerate(contrast) if ind != i]
            if len(b) > 1:
                M = np.mean(temp1[b], axis=0)
            elif len(b) == 1:
                M = temp1[b][0]
            # OPA rotation w.r.t M
            Mp2 = _OPA(M, temp1[levels])
            temp2[levels] = Mp2
        Qm2 = euclidean_distances(temp2.reshape((n_groups, n_dimensions * n_levels)))
        Q = np.tril(Qm1).sum() - np.tril(Qm2).sum()
        Qm1 = Qm2.copy()
        temp1 = temp2.copy()
        iter += 1
    shape_distance = Qm2
    return shape_distance


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
    X = np.matmul(M1.transpose(), M2)
    U, S, Vh = np.linalg.svd(X)
    S = np.diag(S)
    D = np.sign(S)
    V = Vh.transpose()
    H = np.matmul(V, np.matmul(D, U.transpose()))
    Mp2 = np.matmul(M2, H)
    return Mp2


def _get_ls_vectors() -> np.ndarray:
    """
    Generate a least-squares vectors matrix, with intercept and interaction.

    Returns
    -------
    ls_matrix: np.ndarray
        LS vector matrix.
    """
    f_c = [1, 0, 0, 0, 0, 0]
    f_m = [1, 0, 1, 0, 0, 0]
    f_d = [1, 1, 0, 0, 0, 0]
    m_c = [1, 0, 0, 1, 0, 0]
    m_m = [1, 0, 1, 1, 0, 1]
    m_d = [1, 1, 0, 1, 1, 0]
    ls_matrix = np.array([f_c, f_m, f_d, m_c, m_m, m_d])

    return ls_matrix
