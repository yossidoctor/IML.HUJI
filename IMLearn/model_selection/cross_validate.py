from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    ids = np.arange(X.shape[0])
    disjoint_subsets = np.array_split(ids, cv)
    train_score, validation_score = .0, .0
    for i_fold in disjoint_subsets:
        train_mask = ~np.isin(ids, i_fold)
        model = deepcopy(estimator).fit(X[train_mask], y[train_mask])
        train_score += scoring(y[train_mask], model.predict(X[train_mask]))
        validation_score += scoring(y[i_fold], model.predict(X[i_fold]))
    estimated_mean, std = train_score / cv, validation_score / cv
    return estimated_mean, std
