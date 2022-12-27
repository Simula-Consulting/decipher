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
import math
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
)
from urllib.parse import urlparse

from matfact import settings
from matfact.data_generation.dataset import Dataset
from matfact.model import model_factory, prediction_data
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
PLOT_METRICS_W_YLIM = [
    "recall",
    "precision",
    "matthew",
    "accuracy",
]  # Performnace metrics to plot with y_limits
PLOT_METRICS_WO_YLIM = [
    "loss",
    "norm_difference",
]  # Performnace metrics to plot without y_limits
PLOT_ARTIFACTS = [
    "basis",
    "coefs",
    "confusion",
]
NORMAL, INVERTED = "normal", "inverted"  # Dataset types

# Risk state numerical labels
# Normal scenario: # [Normal, LowRisk, HighRisk, Cancer]
# Inverted scenario: # [Cancer, HighRisk, LowRisk, Normal]
NORMAL_LABELS_INT = [1, 2, 3, 4]
INVERTED_LABELS_INT = [4, 3, 2, 1]
LABELS_SHORT_STR = ["N", "LR", "HR", "C"]
LABELS_LONG_STR = ["Normal", "LowRisk", "HighRisk", "Cancer"]


def experiment(
    client,
    run_id,
    hyperparams,
    dataset,
    ax=None,
    labels=NORMAL_LABELS_INT,
    save_run_artifacts=True,
    figure_path: pathlib.Path = FIGURE_PATH,
    display_labels=None,
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
    # # Display confusion matrix for current experiment run
    # if ax is not None:
    #     display_labels = labels if display_labels is None else display_labels
    #     disp = ConfusionMatrixDisplay.from_predictions(
    #         x_true, x_pred, labels=labels, display_labels=display_labels
    #     )
    #     disp.plot(ax=ax, colorbar=False)
    #     # disp.ax_.tick_params("y", labelrotation=90)
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


def matrix_info(m):
    return [
        (int(value), int(count))
        for value, count in np.vstack(np.unique(np.round(m), return_counts=True)).T
    ]


def log_dataset_info(dataset, inverted=False):
    preffix = INVERTED if inverted else NORMAL
    logging.info(f"{preffix} dataset X histogram: {matrix_info(dataset.X)}")
    logging.info(f"{preffix}dataset M histogram: {matrix_info(dataset.M)}")


def invert_domain(X):
    # Inverts matrix label distributions
    return X.max() - (X - X.min())


def invert_dataset(dataset):
    # Inverts dataset label distributions
    log_dataset_info(dataset)
    inv_M = invert_domain(dataset.M.copy())
    inv_X = dataset.X.copy()
    inv_X[inv_X > 0] = invert_domain(inv_X[inv_X > 0])
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
    artifacts_logs, run_logs = [], []
    for run in experiment_runs:
        # Fetch run_id and artifacts path
        run_id = run.info.run_id
        # Fetch flat logs
        run_log = {"run_id": run_id}
        run_log.update(run.data.params)
        run_log.update(run.data.metrics)
        run_logs.append(run_log)
        # Fetch logged artifacts
        artifacts_log = {"run_id": run_id}
        artifacts_log.update(run.data.params)
        artifacts_path = pathlib.Path(urlparse(run.info.artifact_uri).path)
        for artifact in client.list_artifacts(run_id):
            artifact_path = artifacts_path / artifact.path
            artifacts_log[artifact_path.stem] = artifact_path
        client.set_terminated(run_id)
        artifacts_logs.append(artifacts_log)
    # Build dataframe with list of logs and return
    logs_df = pd.DataFrame(run_logs).astype({"lambda1": int, "lambda2": int})
    artifacts_df = pd.DataFrame(artifacts_logs).astype({"lambda1": int, "lambda2": int})
    return logs_df, artifacts_df


def fetch_axis(axs, i, j=None):
    if type(axs) is np.ndarray:
        if j is not None and len(axs.shape) > 1:
            return axs[i, j]
        return axs[i]
    return axs


def preprocess_dataframe(df, metric, filter_lambda, index_lambda, lambda_value):
    # Filter rows for filter_lambda
    filtered_df = df[df[filter_lambda] == lambda_value][[index_lambda, metric]]
    # Sort values by index_lambda and set it as index
    preprocessed_df = filtered_df.sort_values(index_lambda).set_index(index_lambda)
    return preprocessed_df


def plot_metric_against_lambda(
    df, metric, filter_lambda, index_lambda, lambda_value, ax, color
):
    # Plot metrics (y-axis) with different y-resolutions for
    # selected lambda value against index_lambda (x-axis)
    pp_df = preprocess_dataframe(df, metric, filter_lambda, index_lambda, lambda_value)
    ax.plot(pp_df.index, pp_df, color=color, label=f"{index_lambda}: {lambda_value}")


def set_axis_params(ax, x_values, x_label, ylim=None, title=None, legend=True):
    # Sets parameters for the given subplot axes
    if ax is not None:
        y_label = ""
        if legend:
            ax.legend()
        if title is not None:
            ax.set_title(title)
        if ylim is not None:
            y_label = " (shared range)"
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_ylabel(f"metric{y_label}", weight="bold")
        ax.set_xlabel(x_label, weight="bold")
        ax.set_xticks(x_values)
        ax.set_xticklabels(x_values)


def combine_plots(n_lambdas, experiment_name):
    # Create figure to save confusion matrices for all different lambda combinations
    fig_size = (
        4 * (n_lambdas),
        4 * (n_lambdas + 1),
    )  # adapt figure size to number of lambdas
    fig, axs = plt.subplots(n_lambdas, n_lambdas, figsize=fig_size)
    fig.suptitle(experiment_name)


def plot_metrics(
    df_dict,
    metric_name,
    lambda_values,
    fig_name="",
    fig_path=RESULT_PATH,
    y_limits=(-0.05, 1.05),
):
    # Find columns for the given metric
    metric_columns = sorted(
        [col for col in df_dict[NORMAL].columns if metric_name in col]
    )
    # Create list of colors for all metric lines
    colors = pl.cm.viridis(np.linspace(0, 1, len(lambda_values)))
    # Create figure
    fig_cols = 2  # lambda1 and lambda 2 as x-axis
    fig_rows = len(metric_columns) * 2
    fig_size = (15, 5 * len(metric_columns) * 2)
    fig, axs = plt.subplots(fig_rows, fig_cols, figsize=fig_size)
    fig.suptitle(fig_name)
    for i, metric in enumerate(metric_columns):
        # Filter columns
        normal_metric_df = df_dict[NORMAL][[LAMBDA1, LAMBDA2, metric]]
        invert_metric_df = df_dict[INVERTED][[LAMBDA1, LAMBDA2, metric]]
        # Plot line metric
        ax11, ax12 = fetch_axis(axs, i * 2)  # axs[i * 2]
        ax21, ax22 = fetch_axis(axs, i * 2 + 1)  # axs[i * 2 + 1]
        for j, lambda_value in enumerate(lambda_values):
            # select color
            color = colors[j]
            # Plot metrics against lambda1 (normal dataset)
            plot_metric_against_lambda(
                normal_metric_df, metric, LAMBDA2, LAMBDA1, lambda_value, ax11, color
            )
            # Plot metrics against lambda2 (normal dataset)
            plot_metric_against_lambda(
                normal_metric_df, metric, LAMBDA1, LAMBDA2, lambda_value, ax12, color
            )
            # Plot metrics against lambda1 (inverted dataset)
            plot_metric_against_lambda(
                invert_metric_df, metric, LAMBDA2, LAMBDA1, lambda_value, ax21, color
            )
            # Plot metrics against lambda2 (inverted dataset)
            plot_metric_against_lambda(
                invert_metric_df, metric, LAMBDA1, LAMBDA2, lambda_value, ax22, color
            )
        # Set parameters for all subplots
        set_axis_params(
            ax11, lambda_values, LAMBDA1, y_limits, title=f"{metric}_{NORMAL}"
        )
        set_axis_params(
            ax12, lambda_values, LAMBDA2, y_limits, title=f"{metric}_{NORMAL}"
        )
        set_axis_params(
            ax21, lambda_values, LAMBDA1, y_limits, title=f"{metric}_{INVERTED}"
        )
        set_axis_params(
            ax22, lambda_values, LAMBDA2, y_limits, title=f"{metric}_{INVERTED}"
        )

    # Save the figure
    fig.savefig(fig_path / f"{metric_name}_score_{fig_name}.jpg")
    plt.close()


def fetch_lambda_samples(lambda_values, sample_num):
    lambda_samples = []
    if sample_num == len(lambda_values):
        lambda_samples = lambda_values
    else:
        lambda_samples.append(lambda_values[0])
        sample_step_size = round(len(lambda_values) / (sample_num - 1))
        for i in range(sample_num - 2):
            i = (i + 1) * sample_step_size - 1
            lambda_samples.append(lambda_values[i])
        lambda_samples.append(lambda_values[-1])
    return lambda_samples


def fetch_probabilities(lambda_values):
    size = len(lambda_values)
    p = [i**2 for i in range(1, math.floor(size / 2) + 1)] + [
        i**2 for i in range(math.ceil(size / 2), 0, -1)
    ]
    return [i / sum(p) for i in p]


def fetch_lambda_samples(lambda_values, sample_num):
    lambda_samples = []
    if sample_num == len(lambda_values):
        lambda_samples = lambda_values
    else:
        lambda_samples.append(lambda_values[0])
        p = fetch_probabilities(lambda_values[1:-1])
        lambda_samples.extend(
            sorted(
                np.random.choice(
                    lambda_values[1:-1], sample_num - 2, p=p, replace=False
                )
            )
        )
        lambda_samples.append(lambda_values[-1])
    return lambda_samples


def fetch_artifact_path(artifact_df, lambda1, lambda2, artifact_col):
    # Filter rows
    lambda_filter = (artifact_df[LAMBDA1] == lambda1) & (
        artifact_df[LAMBDA2] == lambda2
    )
    return artifact_df[lambda_filter][artifact_col].iat[0]


def fetch_artifact_ax(axs, i, j, inverted):
    if not inverted:
        return fetch_axis(axs, i, j)
    else:
        if len(axs.shape) > 1:
            j = math.ceil(axs.shape[1] / 2) + j
        else:
            i = math.ceil(axs.shape[0] / 2) + i
        return fetch_axis(axs, i, j)


def plot_artifact(ax, image_path, title, inverted=False):
    dataset_type = INVERTED if inverted else NORMAL
    image = plt.imread(image_path)
    ax.imshow(image)
    ax.axis("off")
    ax.set_title(f"{title} {dataset_type}")


def fetch_artifact_column(artifact, df):
    for col in df.columns:
        if artifact in col:
            return col
    logging.info(f"{artifact} was not found in dataframe columns")
    return None


def plot_experiment_artifact(
    logs_dict,
    artifact,
    lambda_values,
    sample_num=3,
    fig_name="",
    fig_path=RESULT_PATH,
):
    artifact_col = fetch_artifact_column(artifact, logs_dict[NORMAL])
    keep_cols = [LAMBDA1, LAMBDA2, artifact_col]
    normal_df, inverted_df = (
        logs_dict[NORMAL][keep_cols],
        logs_dict[INVERTED][keep_cols],
    )
    sample_num = min(sample_num, len(lambda_values))
    lambda_samples = fetch_lambda_samples(lambda_values, sample_num)
    # Create figure
    fig_cols = sample_num * 2  # normal and inverted
    fig_rows = sample_num
    fig_size = (5 * sample_num * 2, 5 * sample_num)
    fig, axs = plt.subplots(fig_rows, fig_cols, figsize=fig_size)
    fig.suptitle(fig_name)

    for i, lambda1 in enumerate(lambda_samples):
        for j, lambda2 in enumerate(lambda_samples):
            title = f"U{lambda1}_V{lambda2}"
            # plot normal artifact
            ax1 = fetch_artifact_ax(axs, i, j, inverted=False)
            image_path1 = fetch_artifact_path(normal_df, lambda1, lambda2, artifact_col)
            plot_artifact(ax1, image_path1, title, inverted=False)

            # plot inverted artifact
            ax2 = fetch_artifact_ax(axs, i, j, inverted=True)
            image_path2 = fetch_artifact_path(
                inverted_df, lambda1, lambda2, artifact_col
            )
            plot_artifact(ax2, image_path2, title, inverted=True)

    fig.savefig(fig_path / f"{artifact}_{fig_name}.jpg")  # , format='svg', dpi=1200)


def run_l2_regularization_experiments(lambda_values, result_path=RESULT_PATH):
    # Creates dataset and run experiment on it and it's inverted labels version.

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
        experiment_name = (
            "exp_" + dataset_type + "_" + datetime.now().strftime("%y%m%d_%H%M%S")
        )
        experiment_id = client.create_experiment(experiment_name)

        # Create figure to save confusion matrices for all different lambda combinations
        n_lambdas = len(lambda_values)  # adapt number of subplots to number of lambdas
        # fig_size = (
        #     4 * (n_lambdas),
        #     4 * (n_lambdas + 1),
        # )  # adapt figure size to number of lambdas
        # fig, axs = plt.subplots(n_lambdas, n_lambdas, figsize=fig_size)
        # fig.suptitle(experiment_name)
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
                # ax = fetch_axis(axs, i, j)  # axs[i][j]
                # ax.set_title(run_name)
                # Run experiment with current hyperparameters
                experiment(
                    client,
                    run_id,
                    hyperparams,
                    dataset,
                    # ax,
                    figure_path=FIGURE_PATH / experiment_id,  # / run_name,
                    labels=labels,
                    display_labels=LABELS_SHORT_STR,
                )

        # # Save confusion matrices figure
        # if settings.create_path_default:
        #     result_path.mkdir(parents=True, exist_ok=True)
        # fig.savefig(result_path / f"conf_matrix_{dataset_type}_{experiment_id}.jpg")
        # plt.close()

        # Retrieve experiment logs and save metric plots figures
        df, artifacts = retrieve_experiment_logs(client, experiment_id)
        run_logs_dfs[dataset_type] = df
        run_artifacts[dataset_type] = artifacts
        experiment_ids.append(experiment_id)
        plot_suffix = "_".join(experiment_ids)
    for artifact in PLOT_ARTIFACTS:
        plot_experiment_artifact(
            run_artifacts, artifact, lambda_values, 3, plot_suffix, result_path
        )

    if n_lambdas > 1:
        pass
        #
        # for metric in PLOT_METRICS_W_YLIM:
        #     plot_metrics(
        #         run_logs_dfs, metric, lambda_values, plot_suffix + "_samey", result_path
        #     )

        # plot_suffix = "_".join(experiment_ids)
        # for metric in PLOT_METRICS_W_YLIM:
        #     plot_metrics(
        #         run_logs_dfs, metric, lambda_values, plot_suffix, result_path, None
        #     )

        # for metric in PLOT_METRICS_WO_YLIM:
        #     plot_metrics(
        #         run_logs_dfs, metric, lambda_values, plot_suffix, result_path, None
        #     )


def main():
    # Define list of lambda values
    lambda_values = [0, 3, 9, 18, 21, 63, 126, 189]
    run_l2_regularization_experiments(lambda_values)


if __name__ == "__main__":
    main()
