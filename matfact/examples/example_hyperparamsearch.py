import functools
from typing import Callable

import mlflow
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

from matfact.data_generation import Dataset
from matfact.model import train_and_log
from matfact.model.logging import MLFlowBatchLogger, MLFlowLogger, dummy_logger_context
from matfact.settings import BASE_PATH, DATASET_PATH


def get_objective(data: Dataset, search_space: list, **hyperparams):
    """Simple train-test based search."""

    X_train, X_test, *_ = data.get_split_X_M()

    @use_named_args(search_space)
    def objective(**search_hyperparams):
        hyperparams.update(search_hyperparams)
        mlflow_output = train_and_log(
            X_train,
            X_test,
            logger_context=MLFlowLogger(),
            dict_to_log=data.prefixed_metadata(),
            log_loss=False,
            **hyperparams,
        )
        # The score logged, the Matthew correlation coefficient, is 'higher is
        # better', while we are minimizing.
        return -mlflow_output["metrics"]["matthew_score"]

    return objective


def get_objective_CV(
    data: Dataset,
    search_space: list,
    n_splits: int = 5,
    log_folds: bool = False,
    **hyperparams,
):
    """Cross validation search."""
    kf = KFold(n_splits=n_splits)
    X, _ = data.get_X_M()
    logger_context = MLFlowLogger() if log_folds else dummy_logger_context

    @use_named_args(search_space)
    def objective(**search_hyperparams):
        hyperparams.update(search_hyperparams)
        scores = []
        with MLFlowBatchLogger() as logger:
            for train_idx, test_idx in kf.split(X):
                mlflow_output = train_and_log(
                    X[train_idx],
                    X[test_idx],
                    dict_to_log=data.prefixed_metadata(),
                    logger_context=logger_context,
                    log_loss=False,
                    **hyperparams,
                )
                # The score logged, the Matthew correlation coefficient, is 'higher is
                # better', while we are minimizing.
                logger(mlflow_output)
                scores.append(-mlflow_output["metrics"]["matthew_score"])
        return np.mean(scores)

    return objective


def example_hyperparameter_search(objective_getter: Callable = get_objective_CV):
    """Example implementation of hyperparameter search.

    objective_getter: callable returning an objective function."""
    tf.config.set_visible_devices([], "GPU")
    mlflow.set_tracking_uri(BASE_PATH / "mlruns")
    space = (
        Real(-5.0, 1, name="lambda1"),
        Real(8, 20, name="lambda2"),
        Real(0.0, 20, name="lambda3"),
    )

    # Load data
    try:
        data = Dataset.from_file(DATASET_PATH)
    except FileNotFoundError:  # No data loaded
        data = Dataset.generate(1000, 40, 5, 5)

    with mlflow.start_run():
        res_gp = gp_minimize(
            objective_getter(
                data, search_space=space, convolution=False, shift_range=None
            ),
            space,
            n_calls=10,
        )

        best_values = res_gp["x"]
        best_score = res_gp["fun"]
        # We are minimizing, so the best_score is inverted.
        mlflow.log_metric("best_score", -best_score)
        for param, value in zip(space, best_values):
            mlflow.log_param(f"best_{param.name}", value)
        mlflow.set_tag("Notes", "Hyperparameter search")


if __name__ == "__main__":
    # Set objective_getter to get_objective_CV to use cross validation.
    # Otherwise, get_objective uses simple train/test split.
    example_hyperparameter_search(
        objective_getter=functools.partial(get_objective_CV, log_folds=True)
    )
