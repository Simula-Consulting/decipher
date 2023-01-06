"""Example using `train_and_log` for simpler workflow.

!!! Warning

    We might change the usage of `train_and_log` in the future.
"""
import pathlib
from itertools import product

import tensorflow as tf

from matfact.data_generation import Dataset
from matfact.model import reconstruction_mse, train_and_log
from matfact.model.logging import MLFlowLoggerDiagnostic
from matfact.settings import settings


def experiment(
    hyperparams,
    enable_shift: bool = False,
    enable_weighting: bool = False,
    enable_convolution: bool = False,
    mlflow_tags: dict | None = None,
    dataset_path: pathlib.Path = settings.paths.dataset,
):
    """Execute and log an experiment.

    Loads the dataset in dataset_path and splits this into train and test sets.
    The test set is masked, so that the last observation is hidden.
    The matrix completion is solved on the train set and then the probability of the
    possible states of the (masked) train set is comptued.

    The run is tracked using MLFLow, using several metrics and artifacts (files).

    Parameters:
        hyperparams: Dict of hyperparameters passed to the model. Common for all models
            is
            {
                rank,
                lambda1,
                lambda2,
            }
        enable_shift: Use shifted model with shift range (-12,12)
        enable_weighting: Use weighted model with weights as defined in
            experiments.simulation::data_weights
        enable_convolution: Use convolutional model.
        mlflow_tags: Dict of tags to mark the MLFLow run with. For example
            {
                "Developer": "Ola Nordmann",
                "Notes": "Using a very slow computer.",
            }
        dataset_path: pathlib.Path The path were dataset is stored.
    """
    # Setup and loading #
    dataset = Dataset.from_file(dataset_path)
    X_train, X_test, M_train, _ = dataset.get_split_X_M()

    shift_range = list(range(-12, 13)) if enable_shift else []

    extra_metrics = {
        "recMSE": lambda model: reconstruction_mse(M_train, model.X, model.M),
    }

    train_and_log(
        X_train,
        X_test,
        shift_range=shift_range,
        use_weights=enable_weighting,
        extra_metrics=extra_metrics,
        use_convolution=enable_convolution,
        logger_context=MLFlowLoggerDiagnostic(
            settings.paths.figure, extra_tags=mlflow_tags
        ),
        **hyperparams
    )


def main():
    # Generate some data
    Dataset.generate(N=1000, T=50, rank=5, sparsity_level=6).save(
        settings.paths.dataset
    )

    USE_GPU = False
    if not USE_GPU:
        tf.config.set_visible_devices([], "GPU")

    mlflow_tags = {
        "Developer": "Thorvald M. Ballestad",
        "GPU": USE_GPU,
        "Notes": "tf.function commented out",
    }
    # NB! lamabda1, lambda2, lambda3 does *not* correspond directly to
    # the notation used in the master thesis.
    hyperparams = {
        "rank": 5,
        "lambda1": -0.021857774198331015,
        "lambda2": 8,
        "lambda3": 4.535681885641427,
    }

    for shift, weight, convolve in product([False, True], repeat=3):
        experiment(
            hyperparams,
            shift,
            weight,
            convolve,
            mlflow_tags=mlflow_tags,
        )


if __name__ == "__main__":
    main()
