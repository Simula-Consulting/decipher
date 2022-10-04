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

This prediction algorithm is implemented in `ClassificationTree`. In addition,
`estimate_probability_thresholds` estimates the optimal values of the thresholds.
"""
from typing import Any

import numpy as np
from scipy import optimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import matthews_corrcoef


class ClassificationTree(BaseEstimator, ClassifierMixin):
    """Perform hierarchical classification given probability thresholds.
    The number of thresholds (tau) is one less than the number of classes.
    """

    def __init__(self, thresholds: np.ndarray | None = None):
        # To adhere to the sklearn API all arguments must have a default value, and
        # we cannot have any logic in the constructor.
        self.thresholds = thresholds

    def predict(self, probabilities: np.ndarray):
        """Perform classification given probabilities for classes.

        Arguments:
        probabilities: (number_of_samples x number_of_states) ndarray
        """

        number_of_samples, number_of_states = probabilities.shape
        if number_of_states != len(self.thresholds) + 1:
            raise ValueError(
                f"Probabilities for {number_of_states} states given. "
                "The number of thresholds should be one less than the number of states"
                f", but it is {len(self.thresholds)}."
            )

        # Set all states to one
        # Iterate through the classes, if it is above the threshold, assign that class.
        states = np.ones(number_of_samples)
        for i, threshold in enumerate(self.thresholds):
            # Threshold i correspond to class i + 1, so add one
            states[probabilities[:, i + 1] >= threshold] = i + 1

        return states

    def fit(self, X: Any, y: Any):
        """Do nothing, fit required by sklearn API specification."""
        return self


def _check_valid_prediction_true_pair(
    true_classes: np.ndarray, probabilities: np.ndarray
):
    """Check that probability predictions and true class labels are compatible.

    Arguments:
    true_classes: (number_of_samples) index of the correct class per sample.
    probabilities: (number_of_samples x number_of_classes) probability of each class
        per sample.

    Raises ValueError on failure."""
    if len(probabilities.shape) != 2:
        raise ValueError(
            "probabilities should be two-dimensional"
            ", one for samples and one for classes."
        )

    if len(true_classes.shape) != 1:
        raise ValueError("true_classes should be one-dimensional.")

    number_of_samples, _ = probabilities.shape
    if number_of_samples != true_classes.size:
        raise ValueError(
            "true_classes and probabilities does not have the same number of samples!"
        )


def mcc_objective(
    thresholds: np.ndarray,
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    clf: ClassificationTree,
):
    "Objective function to evaluate the differential evolution process."
    _check_valid_prediction_true_pair(y_true, y_pred_proba)

    clf.set_params(thresholds=thresholds)

    return -1.0 * matthews_corrcoef(
        y_true.astype(int), clf.predict(y_pred_proba).astype(int)
    )


def estimate_probability_thresholds(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    tol: float = 1e-6,
    seed: int = 42,
    n_thresholds: int = 3,
):
    """Use differential evolution algorithm to estimate probability thresholds for the classification tree.  # noqa: E501

    Args:
        y_true: Vector of class labels.
        y_pred_proba: Vector of predicted probabilities.

    Returns:
        A ClassificationTree object instantiated with the estimated probaility
        thresholds. This object may be saved to disk using scikit-learn routines.
    """

    _check_valid_prediction_true_pair(y_true, y_pred_proba)
    result = optimize.differential_evolution(
        mcc_objective,
        bounds=optimize.Bounds([0] * n_thresholds, [1] * n_thresholds),
        args=(y_true, y_pred_proba, ClassificationTree()),
        seed=np.random.RandomState(seed=seed),
        tol=tol,
    )

    return ClassificationTree(thresholds=result.x)
