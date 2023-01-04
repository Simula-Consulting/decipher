"""
Runs experiment with synthetic data and base matfact on different l2 regularisation parameters for U and V.

Usage:
    ./experiment_l2_regularization.py

Author:
    MQL, Simula Consulting - 2022/12/16
"""

import logging
import pathlib

import numpy as np
import tensorflow as tf


from datetime import datetime
from mlflow import MlflowClient
from sklearn.decomposition import NMF, DictionaryLearning, TruncatedSVD
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    recall_score,
    precision_score,
    confusion_matrix,
)

from matfact import settings
from matfact.data_generation.dataset import Dataset
from matfact.model import model_factory, prediction_data
from matfact.model.factorization.utils import theta_mle
from matfact.model.predict.risk_prediction import predict_proba
from matfact.plotting import (
    plot_basis,
    plot_certainty,
    plot_coefs,
    plot_confusion,
)
from matfact.settings import DATASET_PATH, RESULT_PATH

from .plot_utils import (
    plot_image_artifact,
    plot_numpy_artifact,
    plot_metrics,
    NORMAL,
    INVERTED,
    BASIS,
    COEFS,
    CONFUSION,
    NORMAL_LABELS_INT,
    INVERTED_LABELS_INT,
    LABELS_SHORT_STR,
)
from .experiment_utils import invert_dataset, fetch_experiment_logs

# Set logging level
logging.basicConfig(level=logging.INFO)

MATFACT_ALS = "matfact_als"
SKLEARN_NMF = "sklearn_nmf"
SKLEARN_DL = "sklearn_dl"
SKLEARN_TSVD = "sklearn_tsvd"

# Define global variables for l2 regularization and performance mtrics
PLOT_METRICS_W_YLIM = [
    "recall",
    "precision",
    "matthew",
    "accuracy",
]  # Performnace metrics to plot with y_limits
PLOT_METRICS_WO_YLIM = [
    # "loss",
    # "norm_difference",
]  # Performnace metrics to plot without y_limits
PLOT_IMAGE_ARTIFACTS = []
PLOT_NUMPY_ARTIFACTS = [
    BASIS,
    COEFS,
    CONFUSION,
]


def matfact_fit_predict(
    X_train, X_test_masked, t_pred, hyperparams, U_l1_regularization
):
    # Generate the model
    model = model_factory(
        X_train,
        shift_range=[],
        use_convolution=False,
        use_weights=False,
        U_l1_reg=U_l1_regularization,
        **hyperparams,
    )
    # Train the model (i.e. perform the matrix completion)
    results = model.matrix_completion()
    # Predict the risk over the test set
    p_pred = model.predict_probability(X_test_masked, t_pred)

    return p_pred, results


def sklearn_fit_predict(
    X_train,
    X_test_masked,
    t_pred,
    hyperparams,
    model_type,
):
    # Generate and train the model
    if model_type is SKLEARN_TSVD:
        model = TruncatedSVD(
            n_components=hyperparams["rank"],
            algorithm="randomized",
            n_iter=100,
            random_state=0,
        )
    elif model_type is SKLEARN_DL:
        model = DictionaryLearning(
            n_components=hyperparams["rank"],
            alpha=0,
            positive_code=False,
            positive_dict=False,
            fit_algorithm="cd",
            transform_algorithm="lasso_cd",
            random_state=0,
        )
        # error = model.error_
    elif model_type is SKLEARN_NMF:
        model = NMF(
            n_components=hyperparams["rank"],
            init="random",
            alpha_W=hyperparams["lambda1"],  # U reg
            alpha_H=hyperparams["lambda2"],  # V reg
            l1_ratio=0,
            max_iter=500,
            random_state=0,
        )
        # error = model.reconstruction_err_
    U_train = model.fit_transform(X_train)

    # Store model matrices
    V = model.components_.T  # <- V
    M_train = np.array(U_train @ V.T, dtype=np.float32)

    # Store results
    results = {
        "epochs": [0],
        "error": 0,  # error
        "U": U_train,
        "V": V,
        "M": M_train,
    }

    # Test
    U = model.transform(X_test_masked)  # <- U
    M = np.array(U @ V.T, dtype=np.float32)

    # Predict the risk over the test set
    p_pred = predict_proba(
        X_test_masked, M, t_pred, theta_mle(X_test_masked, M), number_of_states=4
    )

    return p_pred, results


def model_fit_predict(
    X_train,
    X_test_masked,
    t_pred,
    hyperparams,
    model_type=MATFACT_ALS,
    U_l1_regularization=False,
):
    # Predict the risk over the test set
    if "matfact" in model_type:
        p_pred, results = matfact_fit_predict(
            X_train, X_test_masked, t_pred, hyperparams, U_l1_regularization
        )
    elif "sklearn" in model_type:
        p_pred, results = sklearn_fit_predict(
            X_train, X_test_masked, t_pred, hyperparams, model_type
        )

    # Estimate the most likely prediction result from the probabilities
    x_pred = 1.0 + np.argmax(p_pred, axis=1)

    return p_pred, x_pred, results


def experiment(
    client,
    run_id,
    hyperparams,
    dataset,
    labels=NORMAL_LABELS_INT,
    save_numpy_artifacts=True,
    save_image_artifacts=False,
    results_path: pathlib.Path = RESULT_PATH,
    display_labels=LABELS_SHORT_STR,
    model_type=MATFACT_ALS,
    log_matfact_metrics=False,
    U_l1_regularization=False,
):
    """Execute and log an experiment.

    Splits dataset into train and test sets.
    The test set is masked, so that the last observation is hidden.
    The matrix completion is solved on the train set and then the probability of the
    possible states of the (masked) train set is computed.

    The run is tracked with mlflow using client, several metrics and artifacts (files).

    Parameters:
        client: mlflow client
        run_id: mlflow run_id
        hyperparams: Dict of hyperparameters passed to the model.
                     Common for all models: {rank, lambda1, lambda2}
        dataset: dataset containing X, M and metadata
        ax: matplotlib axes to plot run confusion matrix on
    """
    # Log mlflow hyperparameters
    for key, value in hyperparams.items():
        client.log_param(run_id, key, value)
    # Setup and loading
    X_train, X_test, M_train, M_test = dataset.get_split_X_M()
    # Simulate target predictions with the last data point for each sample vetor
    X_test_masked, t_pred, x_true = prediction_data(X_test)
    # Train and Test
    p_pred, x_pred, results = model_fit_predict(
        X_train,
        X_test_masked,
        t_pred,
        hyperparams,
        model_type,
        U_l1_regularization,
    )
    last_step = results["epochs"][-1]
    client.log_metric(
        run_id, "matthew_score", matthews_corrcoef(x_true, x_pred), step=last_step
    )
    client.log_metric(
        run_id, "accuracy", accuracy_score(x_true, x_pred), step=last_step
    )
    precision_scores = precision_score(x_true, x_pred, labels=labels, average=None)
    recall_scores = recall_score(x_true, x_pred, labels=labels, average=None)
    for c, precision, recall in zip(labels, precision_scores, recall_scores):
        client.log_metric(run_id, f"{c}_precision", precision, step=last_step)
        client.log_metric(run_id, f"{c}_recall", recall, step=last_step)
    if "matfact" in model_type and log_matfact_metrics:
        for epoch, loss in zip(results["epochs"], results["loss"]):
            client.log_metric(run_id, "loss", loss, step=epoch)
        client.log_metric(
            run_id, "norm_difference", np.linalg.norm(results["M"] - M_train)
        )

    # Plot and log artifacts for current experiment run
    if save_numpy_artifacts:
        numpy_path = results_path / "numpy"
        if settings.create_path_default:
            numpy_path.mkdir(parents=True, exist_ok=True)
        np.save(numpy_path / "basis.npy", results["V"])
        np.save(numpy_path / "coefs.npy", results["U"])
        np.save(
            numpy_path / "confusion_matrix.npy",
            confusion_matrix(x_true, x_pred, labels=labels),
        )
        client.log_artifacts(run_id, numpy_path)
    if save_image_artifacts and "matfact" in model_type:
        figure_path = results_path / "figures"
        if settings.create_path_default:
            figure_path.mkdir(parents=True, exist_ok=True)
        plot_certainty(p_pred, x_true, figure_path, image_format="jpg")
        plot_coefs(results["U"], figure_path, image_format="jpg")
        plot_basis(results["V"], figure_path, image_format="jpg")
        plot_confusion(
            x_true,
            x_pred,
            figure_path,
            image_format="jpg",
            labels=labels,
            display_labels=display_labels,
        )
        client.log_artifacts(run_id, figure_path)


def run_l2_regularization_experiments(
    lambda_values,
    model_type=MATFACT_ALS,
    result_path=RESULT_PATH,
    experiment_name=None,
    lambda_values_l1=None,
    U_l1_regularization=False,
):
    if lambda_values_l1 is None:
        lambda_values_l1 = lambda_values
    elif len(lambda_values_l1) != len(lambda_values):
        raise ValueError("lambda_values and lambda_values_l1 must have same length!")
    # Creates dataset and run experiment on it and it's inverted labels version.
    if experiment_name is None:
        experiment_name = model_type
    # Generate dataset and invert it
    Dataset.generate(N=10000, T=100, rank=5, sparsity_level=100, censor=False).save(
        DATASET_PATH
    )
    normal_dataset = Dataset.from_file(DATASET_PATH)
    inv_dataset = invert_dataset(normal_dataset)

    # Set GPU use parameters
    USE_GPU = False
    if not USE_GPU:
        tf.config.set_visible_devices([], "GPU")

    # Run experiment on the normal dataset and the inverted dataset
    run_logs_dfs, run_artifacts, experiment_ids = {}, {}, []
    for dataset, dataset_type, labels in [
        (normal_dataset, NORMAL, NORMAL_LABELS_INT),
        (inv_dataset, INVERTED, INVERTED_LABELS_INT),
    ]:
        # Initialize MLFlow client and create experiment for current dataset
        client = MlflowClient()
        exp_name = (
            "exp_"
            + experiment_name
            + "_"
            + dataset_type
            + "_"
            + datetime.now().strftime("%y%m%d_%H%M%S")
        )
        experiment_id = client.create_experiment(exp_name)

        # Create figure to save confusion matrices for all different lambda combinations
        n_lambdas = len(lambda_values)  # adapt number of subplots to number of lambdas
        # Run base matfact for all lambda values (lambda1/U, lambda2/V)
        for lambda1 in lambda_values_l1:
            for lambda2 in lambda_values:
                # Create run
                # run_name = f"U{l2_U}_V{l2_V}"
                run = client.create_run(experiment_id=experiment_id)
                run_id = run.info.run_id
                # Set base matfact hyperparameters for current run
                hyperparams = {
                    "rank": 5,
                    "lambda1": lambda1,
                    "lambda2": lambda2,
                    "lambda3": 0,
                }
                # Run experiment with current hyperparameters
                experiment(
                    client,
                    run_id,
                    hyperparams,
                    dataset,
                    results_path=RESULT_PATH,
                    labels=labels,
                    model_type=model_type,
                    U_l1_regularization=U_l1_regularization,
                )

        # Retrieve experiment logs and save metric plots figures
        df, artifacts = fetch_experiment_logs(client, experiment_id)
        run_logs_dfs[dataset_type] = df
        run_artifacts[dataset_type] = artifacts
        experiment_ids.append(experiment_id)

    plot_suffix = experiment_name + "_" + "_".join(experiment_ids)
    exp_result_path = result_path / plot_suffix
    if settings.create_path_default:
        exp_result_path.mkdir(parents=True, exist_ok=True)

    for artifact in PLOT_IMAGE_ARTIFACTS:
        plot_image_artifact(
            run_artifacts,
            artifact,
            lambda_values,
            3,
            plot_suffix,
            exp_result_path,
            lambda_values_l1=lambda_values_l1,
        )
    for artifact in PLOT_NUMPY_ARTIFACTS:
        plot_numpy_artifact(
            run_artifacts,
            artifact,
            lambda_values,
            3,
            plot_suffix,
            exp_result_path,
            lambda_values_l1=lambda_values_l1,
        )

    if n_lambdas > 1:
        for metric in PLOT_METRICS_W_YLIM:
            plot_metrics(
                run_logs_dfs,
                metric,
                lambda_values,
                plot_suffix + "_samey",
                exp_result_path,
                lambda_values_l1=lambda_values_l1,
            )

        for metric in PLOT_METRICS_W_YLIM:
            plot_metrics(
                run_logs_dfs,
                metric,
                lambda_values,
                plot_suffix,
                exp_result_path,
                None,
                lambda_values_l1=lambda_values_l1,
            )

        for metric in PLOT_METRICS_WO_YLIM:
            plot_metrics(
                run_logs_dfs,
                metric,
                lambda_values,
                plot_suffix,
                exp_result_path,
                None,
                lambda_values_l1=lambda_values_l1,
            )


def main():
    # Define list of lambda values
    lambda_values = [0, 3, 9, 18, 21, 63, 126, 189]
    run_l2_regularization_experiments(lambda_values)


if __name__ == "__main__":
    main()
