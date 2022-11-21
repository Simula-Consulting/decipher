from typing import Protocol

import numpy as np
import numpy.typing as npt

from matfact.model.config import ModelConfig
from matfact.model.factorization import CMF, SCMF, WCMF, BaseMF
from matfact.model.predict.classification_tree import (
    ClassificationTree,
    estimate_probability_thresholds,
)
from matfact.model.predict.dataset_utils import prediction_data


class NotFittedException(Exception):
    pass


def _model_factory(observation_matrix, config: ModelConfig) -> BaseMF:
    if config.shift_budget != [0]:
        return SCMF(observation_matrix, config)
    if config.weight_matrix_getter.is_identity:
        return CMF(observation_matrix, config)
    else:
        return WCMF(observation_matrix, config)


class Predictor(Protocol):
    """Predictor takes probabilities and gives a prediction."""

    def fit(self, matfact, observation_matrix) -> None:
        ...

    def predict(self, probabilities) -> npt.NDArray:
        ...


class ClassificationTreePredictor:
    _classification_tree: ClassificationTree

    def fit(self, matfact, observation_matrix):
        self.matfact = matfact
        self._classification_tree = self._estimate_classification_tree(
            observation_matrix
        )

    def predict(self, probabilities):
        return self._classification_tree.predict(probabilities)

    def _estimate_classification_tree(self, observation_matrix):
        observation_matrix_masked, time_points, true_values = prediction_data(
            observation_matrix
        )
        probabilities = self.matfact.predict_probabilities(
            observation_matrix_masked, time_points
        )
        return estimate_probability_thresholds(true_values, probabilities)


class ArgmaxPredictor:
    def fit(self, matfact, observation_matrix):
        ...

    def predict(self, probabilities):
        return np.argmax(probabilities, axis=1) + 1


class ProbabilityEstimator(Protocol):
    def predict_probability(
        self, matfact, observation_matrix, time_points
    ) -> npt.NDArray:
        ...


class DefaultProbabilityEstimator:
    def predict_probability(
        self, matfact, observation_matrix, time_points
    ) -> npt.NDArray:
        return matfact._factorizer.predict_probability(observation_matrix, time_points)


class MatFact:
    _factorizer: BaseMF
    _predictor: Predictor
    _probability_estimator: ProbabilityEstimator

    def __init__(
        self,
        config: ModelConfig,
        predictor: Predictor | None = None,
        probability_estimator: ProbabilityEstimator | None = None,
        model_factory=_model_factory,
    ):
        """SKLearn like class for MatFact."""
        self._predictor = predictor or ClassificationTreePredictor()
        self._probability_estimator = (
            probability_estimator or DefaultProbabilityEstimator()
        )
        self.config = config
        self._model_factory = model_factory

    def fit(self, observation_matrix):
        """Fit the model."""
        self._factorizer = self._model_factory(observation_matrix, self.config)
        self._factorizer.matrix_completion()
        self._predictor.fit(self, observation_matrix)
        return self

    def predict(self, observation_matrix, time_points):
        probabilities = self.predict_probabilities(observation_matrix, time_points)
        return self._predictor.predict(probabilities)

    def predict_probabilities(self, observation_matrix, time_points):
        self._check_is_fitted()
        return self._probability_estimator.predict_probability(
            self, observation_matrix, time_points
        )

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "_factorizer"):
            raise NotFittedException
