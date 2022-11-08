import pathlib
from itertools import product

import matplotlib
import mlflow
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, matthews_corrcoef

from matfact import settings
from matfact.data_generation import Dataset
from matfact.model import model_factory, prediction_data, reconstruction_mse
from matfact.plotting import (
    plot_basis,
    plot_certainty,
    plot_coefs,
    plot_confusion,
    plot_roc_curve,
)
from matfact.settings import DATASET_PATH, FIGURE_PATH


def experiment(
    hyperparams,
    optimization_params,
    enable_shift: bool = False,
    enable_weighting: bool = False,
    enable_convolution: bool = False,
    mlflow_tags: dict = None,
    dataset_path: pathlib.Path = DATASET_PATH,
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
        optimization_params: Dict passed to the model solver
            {
                num_epochs,
                epochs_per_val,
                patience,
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

    TODO: option to plot and log figures
    TODO: option to store and log artifacts like U,V,M,datasets,etc
    TODO: more clearly separate train and predict
    """
    # Setup and loading #
    dataset = Dataset.from_file(dataset_path)
    X_train, X_test, M_train, M_test = dataset.get_split_X_M()

    # Simulate data for a prediction task by selecting the last data point in each
    # sample vetor as the prediction target
    X_test_masked, t_pred, x_true = prediction_data(X_test, "last_observed")

    mlflow.start_run()
    mlflow.set_tags(mlflow_tags)
    mlflow.log_params(hyperparams)
    mlflow.log_params(optimization_params)
    mlflow.log_params(dataset.prefixed_metadata())

    # Generate the model
    model = model_factory(
        X_train,
        shift_range=list(range(-12, 13)) if enable_shift else [],
        convolution=enable_convolution,
        weights=enable_weighting,
        **hyperparams,
    )

    mlflow.log_param("model_name", model.config.get_short_model_name())

    # Training and testing #
    # Train the model (i.e. perform the matrix completion)
    extra_metrics = {
        "recMSE": lambda model: reconstruction_mse(M_train, X_train, model.M),
    }
    results = model.matrix_completion(
        extra_metrics=extra_metrics, **optimization_params
    )

    # Predict the risk over the test set
    p_pred = model.predict_probability(X_test_masked, t_pred)
    # Estimate the mostl likely prediction result from the probabilities
    x_pred = 1.0 + np.argmax(p_pred, axis=1)

    # We set the backend to have the figure show on Mac.
    # See https://matplotlib.org/stable/users/explain/backends.html for a reference
    # on the matplotlib backends.
    try:
        matplotlib.use("MacOSX")
    except ImportError:  # We are not on a Mac
        pass
    plot_certainty(p_pred, x_true)

    # Log some metrics
    mlflow.log_metric(
        "matthew_score", matthews_corrcoef(x_true, x_pred), step=results["epochs"][-1]
    )
    mlflow.log_metric(
        "accuracy", accuracy_score(x_true, x_pred), step=results["epochs"][-1]
    )
    for epoch, loss in zip(results["epochs"], results["loss"]):
        mlflow.log_metric("loss", loss, step=epoch)

    for metric in extra_metrics:
        for epoch, metric_value in zip(results["epochs"], results[metric]):
            mlflow.log_metric(metric, metric_value, step=epoch)

    mlflow.log_metric("norm_difference", np.linalg.norm(results["M"] - M_train))

    # Plotting #
    if settings.create_path_default:
        FIGURE_PATH.mkdir(parents=True, exist_ok=True)
    plot_coefs(results["U"], FIGURE_PATH)
    plot_basis(results["V"], FIGURE_PATH)
    plot_confusion(x_true, x_pred, FIGURE_PATH)
    plot_roc_curve(x_true, p_pred, FIGURE_PATH)
    mlflow.log_artifacts(FIGURE_PATH)

    mlflow.end_run()


def main():
    # Generate some data
    Dataset.generate(N=1000, T=50, rank=5, sparsity_level=6).save(DATASET_PATH)

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
        "lambda1": 10,
        "lambda2": 10,
        "lambda3": 100,
    }
    optimization_params = {
        "num_epochs": 1000,
        "patience": 5,
    }

    for shift, weight, convolve in product([False, True], repeat=3):
        experiment(
            hyperparams,
            optimization_params,
            shift,
            weight,
            convolve,
            mlflow_tags=mlflow_tags,
        )


if __name__ == "__main__":
    main()
