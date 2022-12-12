from typing import Callable, Protocol, Sequence

import numpy as np
import numpy.typing as npt

from matfact.model.config import ModelConfig
from matfact.model.factorization import CMF, SCMF, WCMF, BaseMF
from matfact.model.predict.classification_tree import (
    ClassificationTree,
    estimate_probability_thresholds,
)
from matfact.model.predict.dataset_utils import prediction_data
from matfact.settings import DEFAULT_AGE_SEGMENTS


class NotFittedException(Exception):
    pass


def _model_factory(
    observation_matrix: npt.NDArray[np.int_], config: ModelConfig
) -> BaseMF:
    if config.shift_budget:
        return SCMF(observation_matrix, config)
    if config.weight_matrix_getter.is_identity:
        return CMF(observation_matrix, config)
    else:
        return WCMF(observation_matrix, config)


class Predictor(Protocol):
    """Predictor takes probabilities and gives a prediction."""

    def fit(self, matfact: "MatFact", observation_matrix: npt.NDArray[np.int_]) -> None:
        ...

    def predict(
        self, probabilities: npt.NDArray[np.float64], time_points: npt.NDArray[np.int_]
    ) -> npt.NDArray[np.int_]:
        ...


class ClassificationTreePredictor:
    """Threshold based predictor.

    Predict class from probabilities using thresholds biasing towards more rare states.
    See [matfact.model.predict.classification_tree][] for details."""

    _classification_tree: ClassificationTree

    def __init__(self, segments: Sequence[int] | None = None) -> None:
        self.segments = DEFAULT_AGE_SEGMENTS if segments is None else segments

    def fit(self, matfact: "MatFact", observation_matrix: npt.NDArray[np.int_]) -> None:
        self.matfact = matfact
        self._classification_tree = self._estimate_classification_tree(
            observation_matrix
        )

    def predict(
        self, probabilities: npt.NDArray[np.float64], time_points: npt.NDArray[np.int_]
    ) -> npt.NDArray[np.int_]:
        age_segment_indexes = self._age_segment_index(time_points)
        return self._classification_tree.predict(probabilities, age_segment_indexes)

    def _age_segment_index(self, time_points: npt.NDArray[np.int_]) -> list[int]:
        """Return the age segment index given time."""
        last_segment_index = len(self.segments)
        # For each time point, going from smaller segment limits,
        # find the first segment limit larger than the time point.
        # In case of no such limits, the correct segment is the last which has no limit.
        return [
            next(
                (i for i, limit in enumerate(self.segments) if time <= limit),
                last_segment_index,
            )
            for time in time_points
        ]

    def _estimate_classification_tree(
        self, observation_matrix: npt.NDArray[np.int_]
    ) -> ClassificationTree:
        """Estimate a ClassificationTree based on the observation data."""
        observation_matrix_masked, time_points, true_values = prediction_data(
            observation_matrix
        )
        probabilities = self.matfact.predict_probabilities(
            observation_matrix_masked, time_points
        )
        age_segment_indexes = self._age_segment_index(time_points)
        number_of_age_segments = len(self.segments) + 1  # Fencepost problem
        thresholds = estimate_probability_thresholds(
            true_values, probabilities, age_segment_indexes, number_of_age_segments
        )
        return ClassificationTree(thresholds)


class ArgmaxPredictor:
    """Maximum probability predictor."""

    def fit(self, matfact: "MatFact", observation_matrix: npt.NDArray[np.int_]) -> None:
        ...

    def predict(
        self, probabilities: npt.NDArray[np.float64], time_points: npt.NDArray[np.int_]
    ) -> npt.NDArray[np.int_]:
        return np.argmax(probabilities, axis=1) + 1


class ProbabilityEstimator(Protocol):
    """Return probabilities for the different states."""

    def predict_probability(
        self,
        matfact: "MatFact",
        observation_matrix: npt.NDArray[np.int_],
        time_points: npt.NDArray[np.int_],
    ) -> npt.NDArray[np.float64]:
        ...


class DefaultProbabilityEstimator:
    def predict_probability(
        self,
        matfact: "MatFact",
        observation_matrix: npt.NDArray[np.int_],
        time_points: npt.NDArray[np.int_],
    ) -> npt.NDArray[np.float64]:
        return matfact._factorizer.predict_probability(observation_matrix, time_points)


class MatFact:
    """SKLearn like class for MatFact."""

    _factorizer: BaseMF

    def __init__(
        self,
        config: ModelConfig,
        predictor: Predictor | None = None,
        probability_estimator: ProbabilityEstimator | None = None,
        model_factory: Callable[
            [npt.NDArray[np.int_], ModelConfig], BaseMF
        ] = _model_factory,
    ) -> None:
        self._predictor = predictor or ClassificationTreePredictor()
        self._probability_estimator = (
            probability_estimator or DefaultProbabilityEstimator()
        )
        self.config = config
        self._model_factory = model_factory

    def fit(self, observation_matrix: npt.NDArray[np.int_]) -> "MatFact":
        """Fit the model."""
        self._factorizer = self._model_factory(observation_matrix, self.config)
        self._factorizer.matrix_completion(epoch_generator=self.config.epoch_generator)
        self._predictor.fit(self, observation_matrix)
        return self

    def predict(
        self,
        observation_matrix: npt.NDArray[np.int_],
        time_points: npt.NDArray[np.int_],
    ) -> npt.NDArray[np.int_]:
        probabilities = self.predict_probabilities(observation_matrix, time_points)
        return self._predictor.predict(probabilities, time_points)

    def predict_probabilities(
        self,
        observation_matrix: npt.NDArray[np.int_],
        time_points: npt.NDArray[np.int_],
    ) -> npt.NDArray[np.float64]:
        self._check_is_fitted()
        return self._probability_estimator.predict_probability(
            self, observation_matrix, time_points
        )

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "_factorizer"):
            raise NotFittedException
