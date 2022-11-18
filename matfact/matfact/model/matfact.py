import numpy as np

from matfact.model.config import ModelConfig
from matfact.model.factorization import SCMF, BaseMF
from matfact.model.factorization.utils import initialize_basis
from matfact.model.predict.classification_tree import (
    ClassificationTree,
    estimate_probability_thresholds,
)
from matfact.model.predict.dataset_utils import prediction_data


class NotFittedException(Exception):
    pass


class MatFact:
    _factorizer: BaseMF
    _classification_tree: ClassificationTree

    def __init__(self, config: ModelConfig):
        self.config = config

    def fit(self, observation_matrix):
        V = initialize_basis(observation_matrix.shape[1], self.config.rank)
        self._factorizer = SCMF(observation_matrix, V, self.config)
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
