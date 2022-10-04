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
import itertools
from typing import Any

import numpy as np
from scipy import optimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import matthews_corrcoef
from sklearn.utils import check_X_y


class ClassificationTree(BaseEstimator, ClassifierMixin):
    """Perform hierarchical classification given probability thresholds.

    Class labels are 1 indexed integers.
    The number of thresholds (tau) is one less than the number of classes.
    """

    def __init__(self, thresholds: np.ndarray | None = None):
        # To adhere to the sklearn API all arguments must have a default value, and
        # we cannot have any logic in the constructor.
        self.thresholds = thresholds

    def predict(self, probabilities: np.ndarray):
        """Perform classification given probabilities for classes.

        Arguments:
        probabilities: (number_of_samples x number_of_classes) ndarray
        """

        number_of_samples, number_of_classes = probabilities.shape
        if number_of_classes != len(self.thresholds) + 1:
            raise ValueError(
                f"Probabilities for {number_of_classes} classes given. "
                "The number of thresholds should be one less than the number of classes"
                f", but it is {len(self.thresholds)}."
            )

        # Set all samples to class one.
        # Iterate through the classes, if the probability is above the
        # threshold, assign that class.
        classes = np.ones(number_of_samples)
        for i, threshold in enumerate(self.thresholds):
            # Threshold i correspond to class i + 1, so add one
            classes[probabilities[:, i + 1] >= threshold] = i + 1

        return classes

    def fit(self, X: Any, y: Any):
        """Do nothing, fit required by sklearn API specification."""
        return self


def _mcc_objective(
    thresholds: np.ndarray,
    y_true: np.ndarray,
    y_predicted_probabilities: np.ndarray,
    clf: ClassificationTree,
):
    "Objective function to evaluate the differential evolution process."
    check_X_y(y_predicted_probabilities, y_true)

    clf.set_params(thresholds=thresholds)

    return -1.0 * matthews_corrcoef(
        y_true.astype(int), clf.predict(y_predicted_probabilities).astype(int)
    )


def estimate_probability_thresholds(
    y_true: np.ndarray,
    y_predicted_probabilities: np.ndarray,
    tol: float = 1e-6,
    seed: int = 42,
):
    """Estimate threshold values for ClassificationTree with differential evolution.

    Args:
        y_true: Vector of class labels.
        y_pred_proba: Vector of predicted probabilities.

    Returns:
        A ClassificationTree object instantiated with the estimated probaility
        thresholds. This object may be saved to disk using scikit-learn routines.
    """
    check_X_y(y_predicted_probabilities, y_true)
    number_of_classes = y_predicted_probabilities.shape[1]

    result = optimize.differential_evolution(
        _mcc_objective,
        # Bounds are [0, 1] for each threshold value, i.e. one less than the number
        # of classes. Iterators are not accepted, so convert to list.
        bounds=list(itertools.repeat((0, 1), number_of_classes - 1)),
        args=(y_true, y_predicted_probabilities, ClassificationTree()),
        seed=seed,
        tol=tol,
    )

    return ClassificationTree(thresholds=result.x)
