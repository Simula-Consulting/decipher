import logging

import numpy as np
from mlflow import MlflowClient

from matfact.data_generation.dataset import Dataset
from matfact.experiments.config import ExperimentConfig, FactorizerType, Hyperparams
from matfact.experiments.sklearn_factorizer import (
    SklearnFactorizer,
    sklearn_model_factory,
)
from matfact.model import model_factory, prediction_data
from matfact.model.factorization.factorizers.cmf import CMF

# Set logging level
logging.basicConfig(level=logging.INFO)


class ExperimentTracker:
    def __init__(self, config: ExperimentConfig, experiment_name: str = None):
        self.config = config
        self._init_mlflow(experiment_name)
        self._init_dataset()
        self._init_hyperparameters()

    def _init_mlflow(self, experiment_name):
        self.client = MlflowClient()
        self.experiment_name = experiment_name or self._generate_experiment_name()
        self.experiment_id = self.client.create_experiment(self.experiment_name)

    def _generate_sexcperiment_name(self):
        # Something based on date?
        pass

    def _init_dataset(self):
        # Generate dataset and invert it
        self.dataset = Dataset.generate(
            N=self.config.N,
            T=self.config.T,
            rank=self.config.rank,
            sparsity_level=self.config.sparsity,
            # censor=self.config.censor,
            # method=self.config.data_gen_method,
        )
        # Setup and loading
        (
            self.X_train,
            self.X_test,
            self.M_train,
            self.M_test,
        ) = self.dataset.get_split_X_M()
        # Simulate target predictions with the last data point for each sample vetor
        self.X_test_masked, self.t_pred, self.x_true = prediction_data(self.X_test)

    def _init_hyperparameters(self):
        self.iterating_hyperparameters = self.config.hyperparameters
        self.hyperparameters = {
            Hyperparams.lambda1: self.config.lambda1,
            Hyperparams.lambda2: self.config.lambda2,
            Hyperparams.rank: self.config.rank,
        }

    def get_factorizer(self, hyperparameters: dict) -> CMF | SklearnFactorizer:
        if self.config.model_type is FactorizerType.DMF:
            return model_factory(
                self.X_train,
                shift_range=[],
                use_convolution=False,
                use_weights=False,
                **hyperparameters,
            )
        else:
            return sklearn_model_factory(self.X_train, **hyperparameters)

    def get_predictions(X_test_masked, t_pred, factorizer):
        # Train the model (i.e. perform the matrix completion)
        results = factorizer.matrix_completion()
        # Predict the risk over the test set
        p_pred = factorizer.predict_probability(X_test_masked, t_pred)
        # Estimate the most likely prediction result from the probabilities
        x_pred = 1.0 + np.argmax(p_pred, axis=1)

        return p_pred, x_pred, results

    def run(self):
        # Run experiment and log while running
        # The experiment has to iterate through the iterating_hyperparameters
        # and log the results under the experiment_id in different runs
        pass

    def plot_metrics(self):
        # Plot the metrics or numpy artifacts in a way that can be compared
        # Save the artifacts in a given path?
        pass
