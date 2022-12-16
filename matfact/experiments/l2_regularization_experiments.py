"""
Runs experiment with synthetic data and base matfact on different l2 regularisation parameters for U and V.

Usage:
    ./experiment_l2_regularization.py

Author:
    MQL, Simula Consulting - 2022/12/16
"""

import logging
import pathlib

import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np
import pandas as pd
import tensorflow as tf


from datetime import datetime
from mlflow import MlflowClient
from mlflow.entities import ViewType
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    recall_score,
    precision_score,
    ConfusionMatrixDisplay,
)

from matfact import settings
from matfact.data_generation.dataset import Dataset
from matfact.model import model_factory, prediction_data, reconstruction_mse
from matfact.plotting import (
    plot_basis,
    plot_certainty,
    plot_coefs,
    plot_confusion,
    plot_roc_curve,
)
from matfact.settings import DATASET_PATH, FIGURE_PATH, RESULT_PATH

# Set logging level
logging.basicConfig(level=logging.INFO)

# Define global variables for l2 regularization and performance mtrics
LAMBDA1, LAMBDA2 = "lambda1", "lambda2"  # U and V regularization parameters
PLOT_METRICS = [
    "recall",
    "precision",
    "matthew",
    "accuracy",
]  # Performnace metrics to plot
NORMAL, INVERTED = "normal", "inverted"

# Risk state numerical labels
# Normal scenario: # [Normal, LowRisk, HighRisk, Cancer]
# Inverted scenario: # [Cancer, HighRisk, LowRisk, Normal]
LABELS = [1, 2, 3, 4]


def experiment(
    client,
    run_id,
    hyperparams,
    dataset,
    ax=None,
    labels=LABELS,
    save_run_artifacts=False,
    figure_path: pathlib.Path = FIGURE_PATH,
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
    # Generate the model
    model = model_factory(
        X_train,
        shift_range=[],
        use_convolution=False,
        use_weights=False,
        **hyperparams,
    )
    # Train the model (i.e. perform the matrix completion)
    results = model.matrix_completion()
    # Predict the risk over the test set
    p_pred = model.predict_probability(X_test_masked, t_pred)
    # Estimate the most likely prediction result from the probabilities
    x_pred = 1.0 + np.argmax(p_pred, axis=1)
    # Display confusion matrix for current experiment run
    if ax is not None:
        disp = ConfusionMatrixDisplay.from_predictions(
            x_true, x_pred, labels=labels, display_labels=labels
        )
        disp.plot(ax=ax, colorbar=False)
    # Log mlflow metrics and parameters for current experiment run
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
    for epoch, loss in zip(results["epochs"], results["loss"]):
        client.log_metric(run_id, "loss", loss, step=epoch)
    client.log_metric(run_id, "norm_difference", np.linalg.norm(results["M"] - M_train))
    # Plot and log artifacts for current experiment run
    if save_run_artifacts:
        if settings.create_path_default:
            figure_path.mkdir(parents=True, exist_ok=True)
            # We set the backend to have the figure show on Mac.
        # try:
        #     matplotlib.use("MacOSX")
        # except ImportError:  # We are not on a Mac
        #     pass
        plot_certainty(p_pred, x_true)
        plot_coefs(results["U"], figure_path)
        plot_basis(results["V"], figure_path)
        plot_confusion(x_true, x_pred, figure_path)
        plot_roc_curve(x_true, p_pred, figure_path)
        client.log_artifacts(run_id, figure_path)


def log_dataset_info(dataset, inverted=False):
    preffix = INVERTED if inverted else ""
    logging.info(f"{preffix}dataset X 0: {sum(dataset.X==0)}")
    logging.info(
        f"{preffix}dataset X histogram: {np.histogram(dataset.X, bins=[0,1,2,3,4,5])}"
    )
    logging.info(
        f"{preffix}dataset M histogram: {np.histogram(dataset.M, bins=[1,2,3,4])}"
    )
    logging.info(f"{preffix}dataset metadata: {dataset.metadata}")


def invert_domain(X, labels):
    # Inverts matrix label distributions
    return X.max() - (X - X.min())


def invert_dataset(dataset, labels):
    # Inverts dataset label distributions
    log_dataset_info(dataset)
    inv_M = invert_domain(dataset.M.copy(), labels)
    inv_X = dataset.X.copy()
    inv_X[inv_X > 0] = invert_domain(inv_X[inv_X > 0], labels)
    inv_metadata = dataset.metadata.copy()
    inv_metadata["observation_probabilities"] = [0.01, 0.04, 0.12, 0.08, 0.03]
    inv_dataset = Dataset(inv_X, inv_M, inv_metadata)
    log_dataset_info(inv_dataset, inverted=True)
    return inv_dataset


def retrieve_experiment_logs(client, experiment_id):
    # Fetch experimet run logs
    experiment_runs = client.search_runs(
        experiment_ids=experiment_id,
        run_view_type=ViewType.ALL,
        order_by=["metric.matthew_score ASC"],
    )

    # Loop through experiment runs and retrieve metrics and artifacts
    artifact_logs, flat_logs = [], []
    for run in experiment_runs:
        # Fetch run_id and artifacts path
        run_logs = {"run_id": run.info.run_id}
        artifacts_path = run.info.artifact_uri + "/"
        # Fetch flat logs
        flat_log = run_logs.copy()
        flat_log.update(run.data.params)
        flat_log.update(run.data.metrics)
        flat_logs.append(flat_log)
        # Fetch logged artifacts
        run_logs["artifacts"] = [
            artifacts_path + artifact.path
            for artifact in client.list_artifacts(run.info.run_id)
        ]
        client.set_terminated(run.info.run_id)
        artifact_logs.append(run_logs)
    # Build dataframe with list of logs and return
    logs_df = pd.DataFrame(flat_logs).astype({"lambda1": int, "lambda2": int})
    return logs_df, artifact_logs


def set_axis_params(ax, x_values, x_label, ylim=None, title=None, legend=True):
    # Sets parameters for the given subplot axes
    y_label = ""
    if legend:
        ax.legend()
    if title is not None:
        ax.set_title(title)
        y_label = " (shared range)"
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_ylabel(f"metric{y_label}", weight="bold")
    ax.set_xlabel(x_label, weight="bold")
    ax.set_xticks(x_values)
    ax.set_xticklabels(x_values)


def preprocess_dataframe(df, metric, filter_lambda, index_lambda, lambda_value):
    filtered_df = df[df[filter_lambda] == lambda_value][[index_lambda, metric]]
    preprocessed_df = filtered_df.sort_values(index_lambda).set_index(index_lambda)
    return preprocessed_df


def plot_metric_against_lambda(ax1, ax2, df, color, index_lambda, lambda_value):
    ax1.plot(df.index, df, color=color, label=f"{index_lambda}_{lambda_value}")
    ax2.plot(df.index, df, color=color, label=f"{index_lambda}_{lambda_value}")


def plot_metrics(
    df_dict, metric_name, lambda_values, fig_name="", result_path=RESULT_PATH
):
    # Find columns for the given metric
    metric_columns = sorted(
        [col for col in df_dict[NORMAL].columns if metric_name in col]
    )
    # Create list of colors for all metric lines
    colors = pl.cm.viridis(np.linspace(0, 1, len(lambda_values)))
    # Create figure
    fig, axs = plt.subplots(
        len(metric_columns) * 2, 4, figsize=(15 * 2, 5 * len(metric_columns) * 2)
    )
    fig.suptitle(fig_name)
    for i, metric in enumerate(metric_columns):
        # Filter columns
        normal_metric_df = df_dict[NORMAL][[LAMBDA1, LAMBDA2, metric]]
        invert_metric_df = df_dict[INVERTED][[LAMBDA1, LAMBDA2, metric]]
        # Plot line metric
        ax11, ax12, ax13, ax14 = axs[i * 2]
        ax21, ax22, ax23, ax24 = axs[i * 2 + 1]
        for j, lambda_value in enumerate(lambda_values):
            # select color
            color = colors[j]
            # Filter rows for lambda2 (V), sort values by lambda1 (U)
            # and set it as index
            filtered_metric = preprocess_dataframe(
                normal_metric_df, metric, LAMBDA2, LAMBDA1, lambda_value
            )
            # Plot metrics (y-axis) for selected lambda2 value against
            # lambda1 (x-axis) with different y-resolutions
            plot_metric_against_lambda(
                ax11, ax12, filtered_metric, color, LAMBDA2, lambda_value
            )
            # Filter rows for lambda1 (U), sort values by lambda2 (V)
            # and set it as index
            filtered_metric = preprocess_dataframe(
                normal_metric_df, metric, LAMBDA1, LAMBDA2, lambda_value
            )
            # Plot metrics (y-axis) for selected lambda1 value against
            # lambda2 (x-axis) with different y-resolutions
            plot_metric_against_lambda(
                ax13, ax14, filtered_metric, color, LAMBDA1, lambda_value
            )
            # Filter rows for lambda2 (V), sort values by lambda1 (U)
            # and set it as index
            filtered_metric = preprocess_dataframe(
                invert_metric_df, metric, LAMBDA2, LAMBDA1, lambda_value
            )
            # Plot metrics (y-axis) for selected lambda2 value against
            # lambda1 (x-axis) with different y-resolutions
            plot_metric_against_lambda(
                ax21, ax22, filtered_metric, color, LAMBDA2, lambda_value
            )
            # Filter rows for lambda1 (U), sort values by lambda2 (V)
            # and set it as index
            filtered_metric = preprocess_dataframe(
                invert_metric_df, metric, LAMBDA1, LAMBDA2, lambda_value
            )
            # Plot metrics (y-axis) for selected lambda1 value against
            # lambda2 (x-axis) with different y-resolutions
            plot_metric_against_lambda(
                ax23, ax24, filtered_metric, color, LAMBDA1, lambda_value
            )

        # Set parameters for all subplots
        set_axis_params(ax11, lambda_values, LAMBDA1, title=f"{metric}_{NORMAL}")
        set_axis_params(ax12, lambda_values, LAMBDA1, (-0.05, 1.05), legend=False)
        set_axis_params(ax13, lambda_values, LAMBDA2, title=f"{metric}_{NORMAL}")
        set_axis_params(ax14, lambda_values, LAMBDA2, (-0.05, 1.05), legend=False)
        set_axis_params(ax21, lambda_values, LAMBDA1, title=f"{metric}_{INVERTED}")
        set_axis_params(ax22, lambda_values, LAMBDA1, (-0.05, 1.05), legend=False)
        set_axis_params(ax23, lambda_values, LAMBDA2, title=f"{metric}_{INVERTED}")
        set_axis_params(ax24, lambda_values, LAMBDA2, (-0.05, 1.05), legend=False)

    # Save the figure
    fig.savefig(result_path / f"{metric_name}_score_{fig_name}.jpg")
    plt.close()


def run_l2_regularization_experiments(
    lambda_values, labels=LABELS, result_path=RESULT_PATH
):
    # Creates dataset and run experiment on it and it's inverted labels version.

    # Generate dataset and invert it
    Dataset.generate(N=10000, T=100, rank=5, sparsity_level=100).save(DATASET_PATH)
    normal_dataset = Dataset.from_file(DATASET_PATH)
    inv_dataset = invert_dataset(normal_dataset, labels)

    # Set GPU use parameters
    USE_GPU = False
    if not USE_GPU:
        tf.config.set_visible_devices([], "GPU")

    # Run experiment on the normal dataset and the inverted dataset
    run_logs_dfs, experiment_ids = {}, []
    for dataset, dataset_type in [(normal_dataset, NORMAL), (inv_dataset, INVERTED)]:
        # Initialize MLFlow client and create experiment for current dataset
        client = MlflowClient()
        experiment_name = (
            "exp_" + dataset_type + "_" + datetime.now().strftime("%y%m%d_%H%M%S")
        )
        experiment_id = client.create_experiment(experiment_name)

        # Create figure to save confusion matrices for all different lambda combinations
        n_lambdas = len(lambda_values)  # adapt number of subplots to number of lambdas
        fig_size = (
            4 * (n_lambdas),
            4 * (n_lambdas + 1),
        )  # adapt figure size to number of lambdas
        fig, axs = plt.subplots(n_lambdas, n_lambdas, figsize=fig_size)
        fig.suptitle(experiment_name)
        # Run base matfact for all lambda values (lambda1/U, lambda2/V)
        for i, l2_U in enumerate(lambda_values):
            for j, l2_V in enumerate(lambda_values):
                # Create run
                run_name = f"U{l2_U}_V{l2_V}"
                run = client.create_run(experiment_id=experiment_id)
                run_id = run.info.run_id
                # Set base matfact hyperparameters for current run
                hyperparams = {
                    "rank": 5,
                    "lambda1": l2_U,
                    "lambda2": l2_V,
                    "lambda3": 0,
                }
                # Rertieve subplot axis and set title
                ax = axs[i][j]
                ax.set_title(run_name)
                # Run experiment with current hyperparameters
                experiment(
                    client,
                    run_id,
                    hyperparams,
                    dataset,
                    ax,
                    figure_path=FIGURE_PATH / experiment_id / run_name,
                    labels=labels,
                )

        # Save confusion matrices figure
        if settings.create_path_default:
            result_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(result_path / f"conf_matrix_{dataset_type}_{experiment_id}.jpg")
        plt.close()

        # Retrieve experiment logs and save metric plots figures
        df, _ = retrieve_experiment_logs(client, experiment_id)
        run_logs_dfs[dataset_type] = df
        experiment_ids.append(experiment_id)

    for metric in PLOT_METRICS:
        plot_metrics(
            run_logs_dfs, metric, lambda_values, "_".join(experiment_ids), result_path
        )


def main():
    # Define list of lambda values
    lambda_values = [0, 3, 9, 18, 21, 63, 126, 189]
    run_l2_regularization_experiments(lambda_values)


if __name__ == "__main__":
    main()
