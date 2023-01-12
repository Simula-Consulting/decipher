import logging
import pathlib

import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np
import math

from sklearn.metrics import ConfusionMatrixDisplay

RESULTS_PATH = pathlib.Path(__file__).parent / "results"

NORMAL, INVERTED = "normal", "inverted"  # Dataset types
LAMBDA1, LAMBDA2 = "lambda1", "lambda2"  # U and V regularization parameters
BASIS, COEFS = "basis", "coefs"
CONFUSION = "confusion"
NUMPY, JPEG = "npy", "jpg"


# Risk state numerical labels
NORMAL_LABELS_INT = [1, 2, 3, 4]
INVERTED_LABELS_INT = [4, 3, 2, 1]
# Risk state string labels
LABELS_SHORT_STR = ["N", "LR", "HR", "C"]
LABELS_LONG_STR = ["Normal", "LowRisk", "HighRisk", "Cancer"]


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
    fig_path=RESULTS_PATH,
    y_limits=(-0.05, 1.05),
    lambda_values_l1=None,
):
    if lambda_values_l1 is None:
        lambda_values_l1 = lambda_values
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
        for j, lambda_values_l1_l2 in enumerate(zip(lambda_values_l1, lambda_values)):
            lambda_value_l1, lambda_value = lambda_values_l1_l2
            # select color
            color = colors[j]
            # Plot metrics against lambda1 (normal dataset)
            plot_metric_against_lambda(
                normal_metric_df, metric, LAMBDA2, LAMBDA1, lambda_value, ax11, color
            )
            # Plot metrics against lambda2 (normal dataset)
            plot_metric_against_lambda(
                normal_metric_df, metric, LAMBDA1, LAMBDA2, lambda_value_l1, ax12, color
            )
            # Plot metrics against lambda1 (inverted dataset)
            plot_metric_against_lambda(
                invert_metric_df, metric, LAMBDA2, LAMBDA1, lambda_value, ax21, color
            )
            # Plot metrics against lambda2 (inverted dataset)
            plot_metric_against_lambda(
                invert_metric_df, metric, LAMBDA1, LAMBDA2, lambda_value_l1, ax22, color
            )
        # Set parameters for all subplots
        set_axis_params(
            ax11, lambda_values_l1, LAMBDA1, y_limits, title=f"{metric}_{NORMAL}"
        )
        set_axis_params(
            ax12, lambda_values, LAMBDA2, y_limits, title=f"{metric}_{NORMAL}"
        )
        set_axis_params(
            ax21, lambda_values_l1, LAMBDA1, y_limits, title=f"{metric}_{INVERTED}"
        )
        set_axis_params(
            ax22, lambda_values, LAMBDA2, y_limits, title=f"{metric}_{INVERTED}"
        )

    # Save the figure
    fig.savefig(fig_path / f"{metric_name}_score_{fig_name}.jpg")
    plt.close()


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


def plot_image(ax, image_path, title, inverted=False):
    dataset_type = INVERTED if inverted else NORMAL
    image = plt.imread(image_path)
    ax.imshow(image)
    ax.axis("off")
    ax.set_title(f"{title} {dataset_type}")


def fetch_artifact_column(artifact, artifact_type, df):
    for col in df.columns:
        if artifact in col and artifact_type in col:
            return col
    logging.info(f"{artifact}.{artifact_type} was not found in dataframe columns")
    return None


def plot_image_artifact(
    logs_dict,
    artifact,
    lambda_values,
    sample_num=3,
    fig_name="",
    fig_path=RESULTS_PATH,
    lambda_values_l1=None,
):
    if lambda_values_l1 is None:
        lambda_values_l1 = lambda_values
    artifact_col = fetch_artifact_column(artifact, JPEG, logs_dict[NORMAL])
    keep_cols = [LAMBDA1, LAMBDA2, artifact_col]
    normal_df, inverted_df = (
        logs_dict[NORMAL][keep_cols],
        logs_dict[INVERTED][keep_cols],
    )
    sample_num = min(sample_num, len(lambda_values))
    lambda_samples = fetch_lambda_samples(lambda_values, sample_num)
    lambda_samples_l1 = fetch_lambda_samples(lambda_values_l1, sample_num)
    # Create figure
    fig_cols = sample_num * 2  # normal and inverted
    fig_rows = sample_num
    fig_size = (5 * sample_num * 2, 5 * sample_num)
    fig, axs = plt.subplots(fig_rows, fig_cols, figsize=fig_size)
    fig.suptitle(fig_name)

    for i, lambda1 in enumerate(lambda_samples_l1):
        for j, lambda2 in enumerate(lambda_samples):
            title = f"U{lambda1}_V{lambda2}"
            # plot normal artifact
            ax1 = fetch_artifact_ax(axs, i, j, inverted=False)
            image_path1 = fetch_artifact_path(normal_df, lambda1, lambda2, artifact_col)
            plot_image(ax1, image_path1, title, inverted=False)

            # plot inverted artifact
            ax2 = fetch_artifact_ax(axs, i, j, inverted=True)
            image_path2 = fetch_artifact_path(
                inverted_df, lambda1, lambda2, artifact_col
            )
            plot_image(ax2, image_path2, title, inverted=True)

    fig.savefig(fig_path / f"{artifact}_{fig_name}.jpg")  # , format='svg', dpi=1200)


def plot_confusion_artifact(ax, confusion_path, display_labels, title, inverted=False):
    "Make a histogram of coefficients from the U matrix."
    dataset_type = INVERTED if inverted else NORMAL
    conf_mat = np.load(confusion_path)
    disp = ConfusionMatrixDisplay(conf_mat, display_labels=display_labels)
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"{title} {dataset_type}")


def plot_coefs_artifact(ax, u_path, title, inverted=False, n_bins=50):
    "Make a histogram of coefficients from the U matrix."
    dataset_type = INVERTED if inverted else NORMAL
    U = np.load(u_path)
    hist, bins = np.histogram(U.ravel(), bins=n_bins)
    ax.bar((bins[:-1] + bins[1:]) / 2, hist)
    ax.set_ylabel("Count")
    ax.set_xlabel("Coefficient")
    ax.set_title(f"{title} {dataset_type}")


def plot_basis_artifact(ax, v_path, title, inverted=False):
    "Plot the basic vectors in the V matrix."
    dataset_type = INVERTED if inverted else NORMAL
    V = np.load(v_path)
    ax.plot(V)
    ax.set_xlabel("Column coordinate")
    ax.set_ylabel("Basic vector")
    ax.set_title(f"{title} {dataset_type}")


def plot_numpy_artifact(
    logs_dict,
    artifact,
    lambda_values,
    sample_num=3,
    fig_name="",
    fig_path=RESULTS_PATH,
    display_labels=LABELS_SHORT_STR,
    lambda_values_l1=None,
):
    if lambda_values_l1 is None:
        lambda_values_l1 = lambda_values
    artifact_col = fetch_artifact_column(artifact, NUMPY, logs_dict[NORMAL])
    keep_cols = [LAMBDA1, LAMBDA2, artifact_col]
    normal_df, inverted_df = (
        logs_dict[NORMAL][keep_cols],
        logs_dict[INVERTED][keep_cols],
    )
    sample_num = min(sample_num, len(lambda_values))
    lambda_samples = fetch_lambda_samples(lambda_values, sample_num)
    lambda_samples_l1 = fetch_lambda_samples(lambda_values_l1, sample_num)
    # Create figure
    fig_cols = sample_num * 2  # normal and inverted
    fig_rows = sample_num
    fig_size = (5 * sample_num * 2, 5 * sample_num)
    fig, axs = plt.subplots(
        fig_rows, fig_cols, sharex="all", sharey="all", figsize=fig_size
    )
    fig.suptitle(fig_name)

    for i, lambda1 in enumerate(lambda_samples_l1):
        for j, lambda2 in enumerate(lambda_samples):
            title = f"U{lambda1}_V{lambda2}"
            # plot normal and inverted artifact
            ax1 = fetch_artifact_ax(axs, i, j, inverted=False)
            array_path1 = fetch_artifact_path(normal_df, lambda1, lambda2, artifact_col)
            ax2 = fetch_artifact_ax(axs, i, j, inverted=True)
            array_path2 = fetch_artifact_path(
                inverted_df, lambda1, lambda2, artifact_col
            )
            if artifact is BASIS:
                plot_basis_artifact(ax1, array_path1, title, inverted=False)
                plot_basis_artifact(ax2, array_path2, title, inverted=False)
            elif artifact is COEFS:
                plot_coefs_artifact(ax1, array_path1, title, inverted=False)
                plot_coefs_artifact(ax2, array_path2, title, inverted=True)
            elif artifact is CONFUSION:
                plot_confusion_artifact(
                    ax1, array_path1, display_labels, title, inverted=False
                )
                plot_confusion_artifact(
                    ax2, array_path2, display_labels, title, inverted=True
                )

    fig.savefig(fig_path / f"{artifact}_{fig_name}.jpg")  # , format='svg', dpi=1200)
