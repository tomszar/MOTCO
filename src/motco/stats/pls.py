"""Partial Least Squares utilities (PLS-DA, VIP computation).

This module provides:
- `plsda_doubleCV`: double cross-validation over number of components
  for PLS-DA classification using AUROC as criterion.
- `_calculate_vips`: compute Variable Importance in Projection (VIP) scores
  from a fitted `sklearn.cross_decomposition.PLSRegression` model.
"""

import multiprocessing
from itertools import repeat
from typing import Union

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder


def plsda_doubleCV(
    X: pd.DataFrame,
    y: Union[pd.DataFrame, pd.Series],
    cv1_splits: int = 7,
    cv2_splits: int = 8,
    n_repeats: int = 30,
    max_components: int = 50,
    random_state: int = 1203,
) -> dict[str, Union[PLSRegression, pd.DataFrame]]:
    """
    Estimate a double cross validation on a partial least squares
    regression - discriminant analysis.

    Parameters
    ----------
    X: pd.DataFrame
        The predictor variables.
    y: Union[pd.DataFrame, pd.Series]
        The outcome varibale.
    cv1_splits: int
        Number of folds in the CV1 loop. Default: 7.
    cv2_splits: int
        Number of folds in the CV2 loop. Default: 8.
    n_repeats: int
        Number of repeats to the cv2 procedure. Default: 30.
    max_components: int
        Maximum number of LV to test. Default: 50.
    random_state: int
        For reproducibility. Default: 1203.

    Returns
    -------
    models_table: dict[str,
                       Union[PLSRegression,
                             pd.DataFrame]
        Dictionary with the table of the best models, including repetition,
        number of latent variables, and AUROC. Also includes the model for
        prediction.
    """
    encoder = OneHotEncoder(sparse_output=False)
    yd = pd.DataFrame(encoder.fit_transform(np.array(y).reshape(-1, 1)))
    cv2 = RepeatedStratifiedKFold(
        n_splits=cv2_splits, n_repeats=n_repeats, random_state=random_state
    )
    cv1 = StratifiedKFold(n_splits=cv1_splits)
    cv2_table = pd.DataFrame(np.zeros((cv2_splits, 2)))
    cv1_table = pd.DataFrame(np.zeros((cv1_splits, 2)))
    for_table = {
        "rep": list(range(1, n_repeats + 1)),
        "LV": list(range(1, n_repeats + 1)),
        "AUROC": [0.1] * n_repeats,
    }
    model_table = pd.DataFrame(for_table)
    row_cv2 = 0
    row_model_table = 0
    cv2_models = []
    best_models = []
    for rest, test in cv2.split(X, y):
        # Outer CV2 loop split into test and rest
        X_rest = X.iloc[rest, :]
        X_test = X.iloc[test, :]
        y_rest = y.iloc[rest]
        yd_rest = yd.iloc[rest, :]
        yd_test = yd.iloc[test, :]
        row_cv1 = 0
        for train, validation in cv1.split(X_rest, y_rest):
            # Inner CV validates optimal number of LVs
            X_train = X_rest.iloc[train, :]
            yd_train = yd_rest.iloc[train, :]
            X_val = X_rest.iloc[validation, :]
            yd_val = yd_rest.iloc[validation, :]
            ns = list(range(1, max_components))
            with multiprocessing.Pool(processes=None) as pool:
                auroc = pool.starmap(
                    _plsda_auroc,
                    zip(
                        ns,
                        repeat(X_train),
                        repeat(yd_train),
                        repeat(X_val),
                        repeat(yd_val),
                    ),
                )
            nlv = auroc.index(max(auroc)) + 1
            cv1_table.iloc[row_cv1, 0] = nlv
            cv1_table.iloc[row_cv1, 1] = max(auroc)
            row_cv1 += 1
        # Obtain optimal n of components
        n_components = int(cv1_table.iloc[cv1_table[1].idxmax(), 0])
        model_score = _plsda_auroc(
            n_components, X_rest, yd_rest, X_test, yd_test, return_full=True
        )
        cv2_table.iloc[row_cv2, 0] = n_components
        cv2_table.iloc[row_cv2, 1] = model_score["score"]
        cv2_models.append(model_score["model"])
        row_cv2 += 1
        if row_cv2 == cv2_splits:
            best_cv2_lv = int(cv2_table.iloc[cv2_table[1].idxmax(), 0])
            auroc_val = cv2_table.iloc[cv2_table[1].idxmax(), 1]
            model_table.iloc[row_model_table, 1] = best_cv2_lv
            model_table.iloc[row_model_table, 2] = auroc_val
            best_models.append(cv2_models[cv2_table[1].idxmax()])
            row_model_table += 1
            cv2_table = pd.DataFrame(np.zeros((cv2_splits, 2)))
            cv2_models = []
            row_cv2 = 0
    models_table = {"models": best_models, "table": model_table}
    return models_table


def _plsda_auroc(
    n_components: int,
    X_train: pd.DataFrame,
    Y_train: Union[pd.Series, pd.DataFrame],
    X_test: pd.DataFrame,
    Y_test: Union[pd.Series, pd.DataFrame],
    return_full: bool = False,
) -> Union[float, dict[str, Union[PLSRegression, float]]]:
    """
    Estimate a partial least squares regression and return the AUROC value
    and the model, or just the AUROC value.

    Parameters
    ----------
    n_components: int
        Number of components to use.
    X_train: pd.DataFrame
        The predictors to use for training.
    Y_train: Union[pd.Series, pd.DataFrame]
        The outcome to use for training.
    X_test: pd.DataFrame,
        The predictors to use for testing.
    Y_test: Union[pd.Series, pd.DataFrame]
        The outcomes to use for testing.
    return_full: bool
        Whether to return the model and the auroc score.
        If False, returns only the auroc score. Default: False.

    Returns
    -------
    auroc: Union[float,
                 dict[str,
                      Union[PLSRegression,
                            float]]]
        Return the auroc score, and optionally the regression model.
    """
    pls = PLSRegression(n_components=n_components, scale=True, max_iter=1000).fit(
        X=X_train, y=Y_train
    )
    y_pred = pls.predict(X_test)
    score = roc_auc_score(Y_test, y_pred)
    if return_full:
        auroc = {"model": pls, "score": score}
    else:
        auroc = score
    return auroc


def _calculate_vips(
    model,
    components: Union[None, list[int]] = None,
) -> np.ndarray:
    """
    Estimates Variable Importance in Projection (VIP)
    in Partial Least Squares (PLS)

    Parameters
    ----------
    model: PLSRegression
        model generated from the PLSRegression function
    components: Union[None, list[int]]
        if not None, a list of integers indicating the components to compute
        the VIPs from. If None, all components are taken into account.
        Default None.

    Returns
    -------
    vips: np.array
        variable importance in projection for each variable
    """
    if components is not None:
        t = model.x_scores_[:, components]
        w = model.x_weights_[:, components]
        q = model.y_loadings_[:, components]
        p, h = w.shape
    else:
        w = model.x_weights_
        t = model.x_scores_
        q = model.y_loadings_
        p, h = w.shape

    vips = np.zeros((p,))
    s = np.diag(np.matmul(np.matmul(np.matmul(t.T, t), q.T), q)).reshape(h, -1)
    total_s = np.sum(s)
    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j])) ** 2 for j in range(h)])
        vips[i] = np.sqrt(p * (np.matmul(s.T, weight)) / total_s)
    return vips