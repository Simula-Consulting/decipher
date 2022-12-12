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
Furthermore, we segment the population into age segments, and perform the above
individually per segment.
"""

import itertools
import random
from enum import Enum, auto
from typing import Any, Sequence

import numpy as np
from scipy import optimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import matthews_corrcoef
from sklearn.utils import check_X_y


class ClassificationTree(BaseEstimator, ClassifierMixin):
    """Perform hierarchical classification given probability thresholds.

    Class labels are 1 indexed integers.
    The number of thresholds (tau) is one less than the number of classes.

    Arguments:
        thresholds: (number_of_age_segments x [number_of_classes - 1])
    """

    def __init__(self, thresholds: Sequence[Sequence[float]] | None = None):
        self.thresholds = thresholds if thresholds is not None else [[]]

    def predict(self, probabilities: np.ndarray, age_segments: Sequence[int]):
        """Perform classification given probabilities for classes.

        Arguments:
            probabilities: (number_of_samples x number_of_classes)
            age_segments: (number_of_samples) age segment index of individual
        """

        number_of_samples, number_of_classes = probabilities.shape
        if number_of_classes != len(self.thresholds[0]) + 1:
            raise ValueError(
                f"Probabilities for {number_of_classes} classes given. "
                "The number of thresholds should be one less than the number of classes"
                f", but it is {len(self.thresholds[0])}."
            )

        if number_of_samples != len(age_segments):
            raise ValueError("Supply the age segment for each individual!")

        # Set all samples to class one.
        # Iterate through the classes, if the probability is above the
        # threshold, assign that class.
        classes = np.ones(number_of_samples, dtype=int)

        # Starting from the second class
        for class_index in range(1, number_of_classes):
            try:
                thresholds_class = [
                    self.thresholds[i][class_index - 1] for i in age_segments
                ]
            except IndexError as e:
                raise ValueError(
                    "Encountered an age segment for which there are not thresholds!"
                ) from e

            classes[probabilities[:, class_index] >= thresholds_class] = (
                class_index + 1  # Compensate for 1-indexed labels.
            )
        return classes

    def fit(self, X: Any, y: Any):
        """Do nothing, fit required by sklearn API specification."""
        return self


def _init_from_partitions(probabilities: np.ndarray, population_size: int = 15):
    """Generate initial samples from the partitions defined by probabilities.

    For each state, we need only test taking the threshold equal to the
    probabilities observed. Here, we draw a population_size sample
    from all possibilities."""
    # Per class, these are the 'interesting' threshold limits
    threshold_limits = ((*set(p), 1) for p in probabilities.T[1:])
    # Sample from all possible points
    population = list(itertools.product(*threshold_limits))

    # If the population is smaller than the requested population_size, sample without
    # replacement will fail. Therefore, return the entire population + random choices
    # (with replacement) from the population as padding.
    if len(population) < population_size:
        return population + random.choices(
            population, k=population_size - len(population)
        )

    return random.sample(
        population,
        k=population_size,
    )


class ThresholdInitMethod(Enum):
    """Init method to use for threshold estimation."""

    DEFAULT = auto()
    """Latin hypercube. Initialize with points drawn from evenly spaced partitions."""
    PARTITION = auto()
    """Partition the threshold space using the observed probabilities.

    This has the advantage of never selecting thresholds that fall in the
    same probability partition."""


def estimate_probability_thresholds(
    y_true: np.ndarray,
    y_predicted_probabilities: np.ndarray,
    age_segments: list[int],
    number_of_age_segments: int,
    init_method: ThresholdInitMethod = ThresholdInitMethod.DEFAULT,
    tol: float = 1e-6,
    seed: int = 42,
):
    """Estimate threshold values for ClassificationTree with differential evolution.

    Returns:
        A ClassificationTree object instantiated with the estimated probability
            thresholds.
    """
    check_X_y(y_predicted_probabilities, y_true)
    number_of_classes = y_predicted_probabilities.shape[1]

    def _matthews_correlation_coefficient_objective(
        thresholds: np.ndarray,
        y_true: np.ndarray,
        y_predicted_probabilities: np.ndarray,
        age_segments: list[int],
        clf: ClassificationTree,
    ) -> float:
        "Objective function to evaluate the differential evolution process."
        thresholds = thresholds.reshape((number_of_age_segments, number_of_classes - 1))
        clf.set_params(thresholds=thresholds)

        return -1.0 * matthews_corrcoef(
            y_true, clf.predict(y_predicted_probabilities, age_segments)
        )

    init: str | list
    match init_method:
        case ThresholdInitMethod.DEFAULT:
            init = "latinhypercube"
        case ThresholdInitMethod.PARTITION:
            init_single_segment = _init_from_partitions(y_predicted_probabilities)
            init = [
                thresholds * number_of_age_segments
                for thresholds in init_single_segment
            ]
        case _:
            raise ValueError(f"Unknown init method {init_method}.")

    result = optimize.differential_evolution(
        _matthews_correlation_coefficient_objective,
        # Bounds are [0, 1] for each threshold value, i.e. one less than the number
        # of classes. Iterators are not accepted, so convert to list.
        bounds=[(0, 1)] * (number_of_classes - 1) * number_of_age_segments,
        args=(
            y_true,
            y_predicted_probabilities,
            age_segments,
            ClassificationTree(),
        ),
        seed=seed,
        tol=tol,
        init=init,  # type: ignore  # Init may be either str or list of values
    )
    thresholds = result.x.reshape((number_of_age_segments, number_of_classes - 1))
    return ClassificationTree(thresholds=thresholds)
