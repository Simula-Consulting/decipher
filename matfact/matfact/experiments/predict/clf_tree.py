import copy

import numpy as np
from scipy import optimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import matthews_corrcoef


class ClassificationTree(BaseEstimator, ClassifierMixin):
    """Perform hierarchical classification given probability thresholds.
    The number of thresholds (tau) equals is one less than the number of classes.
    """

    def __init__(self, tau2=0, tau3=0):

        self.tau2 = tau2
        self.tau3 = tau3

    def get_params(self, deep=True):

        params = {"tau2": self.tau2, "tau3": self.tau3}

        if deep:
            for key in params.keys():
                params[key] = copy.deepcopy(params[key])

        return params

    def set_params(self, **params):

        self.tau2 = params["tau2"]
        self.tau3 = params["tau3"]

        return self

    def predict(self, proba):

        # Bottom-up.
        states = np.ones(proba.shape[0], dtype=float)
        states[proba[:, 1] >= self.tau2] = 2
        states[proba[:, 2] >= self.tau3] = 3

        return states

    def fit(self, X, y):
        # Only for API consistency.
        return self


def mcc_objective(thresh, y_true, y_pred_proba, clf):
    "Objective function to evaluate the differential evolution process."

    clf.set_params(tau2=thresh[0], tau3=thresh[1])

    return -1.0 * matthews_corrcoef(
        y_true.astype(int), clf.predict(y_pred_proba).astype(int)
    )


def estimate_proba_thresh(y_true, y_pred_proba, tol=1e-6, seed=42, n_thresholds=2):
    """Use differential evolution algorithm to estimate probability thresholds for the classification tree.  # noqa: E501

    Aags:
        y_true: Vector of class labels.
        y_pred_proba: Vector of predicted probabilities.

    Returns:
        A ClassificationTree objecti instantiated with the estimated probaility
        thresholds. This object may be saved to disk using scikit-learn routines.
    """

    result = optimize.differential_evolution(
        mcc_objective,
        bounds=optimize.Bounds([0] * n_thresholds, [1] * n_thresholds),
        args=(y_true, y_pred_proba, ClassificationTree()),
        seed=np.random.RandomState(seed=seed),
        tol=tol,
    )

    return ClassificationTree(*result.x)
