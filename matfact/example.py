import pathlib
from itertools import product

import mlflow
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, matthews_corrcoef

from data_generation.main import Dataset
from experiments.algorithms.optimization import matrix_completion
from experiments.algorithms.risk_prediction import predict_proba
from experiments.algorithms.utils import reconstruction_mse
from experiments.main import model_factory
from experiments.plotting.diagnostic import (
    plot_basis,
    plot_coefs,
    plot_confusion,
    plot_roc_curve,
)
from experiments.simulation import data_weights, prediction_data
from settings import DATASET_PATH, FIGURE_PATH


def experiment(
    hyperparams,
    optimization_params,
    enable_shift: bool = False,
    enable_weighting: bool = False,
    enable_convolution: bool = False,
    mlflow_tags: dict = None,
    dataset_path: pathlib.Path = DATASET_PATH,
):
    """
    hyperparams = {
        rank,
        lambda1,
        lambda2,
    }
    optimization_params = {
        num_epochs,
        epochs_per_val,
        patience,
    }

    TODO: option to plot and log figures
    TODO: option to store and log artifacts like U,V,M,datasets,etc
    TODO: more clearly separate train and predict
    """
    # Setup and loading #
    dataset = Dataset().load(dataset_path)
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
    model_name, model = model_factory(
        X_train,
        shift_range=np.arange(-12, 13) if enable_shift else np.array([]),
        convolution=enable_convolution,
        weights=data_weights(X_train) if enable_weighting else None,
        **hyperparams,
    )

    mlflow.log_param("model_name", model_name)

    # Training and testing #
    # Train the model (i.e. perform the matrix completion)
    extra_metrics = (
        ("recMSE", lambda model: reconstruction_mse(M_train, X_train, model.M)),
    )
    results = matrix_completion(
        model, X_train, extra_metrics=extra_metrics, **optimization_params
    )

    # Predict the risk over the test set using the results from matrix completion as
    # input parameters to the prediction algorithm
    p_pred = predict_proba(X_test_masked, results["M"], t_pred, results["theta_mle"])
    # Estimate the mostl likely prediction result from the probabilities
    x_pred = 1.0 + np.argmax(p_pred, axis=1)

    # Log some metrics
    mlflow.log_metric(
        "matthew_score", matthews_corrcoef(x_true, x_pred), step=results["epochs"][-1]
    )
    mlflow.log_metric(
        "accuracy", accuracy_score(x_true, x_pred), step=results["epochs"][-1]
    )
    for epoch, loss in zip(results["epochs"], results["loss_values"]):
        mlflow.log_metric("loss", loss, step=epoch)

    for metric, _ in extra_metrics:
        for epoch, metric_value in zip(results["epochs"], results[metric]):
            mlflow.log_metric(metric, metric_value, step=epoch)

    mlflow.log_metric("norm_difference", np.linalg.norm(results["M"] - M_train))

    # Plotting #
    plot_coefs(results["U"], FIGURE_PATH)
    plot_basis(results["V"], FIGURE_PATH)
    plot_confusion(x_true, x_pred, FIGURE_PATH)
    plot_roc_curve(x_true, p_pred, FIGURE_PATH)
    mlflow.log_artifacts(FIGURE_PATH)

    mlflow.end_run()


def main():
    # Generate some data
    Dataset().generate(N=1000, T=50, rank=5, sparsity_level=6).save(DATASET_PATH)

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