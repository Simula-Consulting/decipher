"""Threshold based classification.

To correct for the data being skewed, we introduce some threshold values that
favor less likely classes, even when they do not have the highest probability.
Consider we have some probabilities [p0, p1, p2], where pi is the probability of
class i. We introduce thresholds [t1, t2], such that instead of choosing the class
with the highest probability, we choose the class as follows: beginning from the class
with the highest number (assumed to be the rearest), check if p2 > t2. If so, we set
the class to p2. Next, check p1 > t1, etc.

In general, given probabilities [p0, p1, p2, ...] and thresholds [t1, t2, ...], set the
class to max(i) where pi > ti.

This prediction algorithm is implemented in `ClassificationTree`.
In addition, `estimate_proba_thresh` estimates the optimal values of the threshodls.
"""
import copy

import numpy as np
from scipy import optimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import matthews_corrcoef


class ClassificationTree(BaseEstimator, ClassifierMixin):
    """Perform hierarchical classification given probability thresholds.
    The number of thresholds (tau) is one less than the number of classes.
    """

    def __init__(self, tau2=0, tau3=0):

        self.tau2 = tau2
        self.tau3 = tau3

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

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
