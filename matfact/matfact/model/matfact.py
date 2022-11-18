import numpy as np

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


class MatFact:
    _factorizer: BaseMF
    _classification_tree: ClassificationTree

    def __init__(self, config: ModelConfig, model_factory=_model_factory):
        """SKLearn like class for MatFact."""
        self.config = config
        self._model_factory = model_factory

    def fit(self, observation_matrix):
        """Fit the model."""
        self._factorizer = self._model_factory(observation_matrix, self.config)
        self._factorizer.matrix_completion()
        if self.config.use_threshold_optimization:
            self._classification_tree = self._estimate_classification_tree(
                observation_matrix
            )
        return self

    def predict(self, observation_matrix, time_points):
        probabilities = self.predict_probabilities(observation_matrix, time_points)
        predict_func = (
            self._classification_tree.predict
            if self.config.use_threshold_optimization
            else lambda probabilities: np.argmax(probabilities, axis=1) + 1
        )
        return predict_func(probabilities)

    def predict_probabilities(self, observation_matrix, time_points):
        self._check_is_fitted()
        return self._factorizer.predict_probability(observation_matrix, time_points)

    def _estimate_classification_tree(self, observation_matrix):
        observation_matrix_masked, time_points, true_values = prediction_data(
            observation_matrix
        )
        probabilities = self.predict_probabilities(
            observation_matrix_masked, time_points
        )
        return estimate_probability_thresholds(true_values, probabilities)

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "_factorizer"):
            raise NotFittedException
