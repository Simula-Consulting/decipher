import pathlib
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

from matfact.settings import settings

from . import plot_config, plot_utils

plot_config.setup()


def plot_coefs(U, path_to_figure: pathlib.Path, fname="", n_bins=50):
    "Make a histogram of coefficients from the U matrix."

    hist, bins = np.histogram(U.ravel(), bins=n_bins)

    fig = plt.figure(figsize=plot_utils.set_fig_size(430, fraction=1, subplots=(1, 1)))
    axis = fig.gca()
    axis.bar((bins[:-1] + bins[1:]) / 2, hist)
    axis.set_ylabel("Count")
    axis.set_xlabel("Coefficient")

    plot_utils.set_arrowed_spines(fig, axis)

    fig.tight_layout()
    fig.savefig(
        path_to_figure / f"coefs_{fname}.pdf", transparent=True, bbox_inches="tight"
    )
    plt.close()


def plot_basis(V, path_to_figure: pathlib.Path, fname=""):
    "Plot the basic vectors in the V matrix."

    fig = plt.figure(figsize=plot_utils.set_fig_size(430, fraction=1, subplots=(1, 1)))
    axis = fig.gca()
    axis.plot(V)
    axis.set_xlabel("Column coordinate")
    axis.set_ylabel("Basic vector")

    plot_utils.set_arrowed_spines(fig, axis)

    fig.tight_layout()
    fig.savefig(
        path_to_figure / f"basis_{fname}.pdf", transparent=True, bbox_inches="tight"
    )
    plt.close()


def _confusion(true, pred, n_classes=4):
    # Auxillary function producing a confusion matrix.

    cmat = np.zeros((n_classes, n_classes), dtype=int)
    for i, x in enumerate(true):

        cmat[int(x - 1), int(pred[i] - 1)] += 1

    return cmat


def plot_confusion(x_true, x_pred, path_to_figure: pathlib.Path, n_classes=4, fname=""):
    "PLot a confusion matrix to compare predictions and ground truths."

    cmat = _confusion(x_true, x_pred, n_classes=n_classes)

    fig, axis = plt.subplots(
        1, 1, figsize=plot_utils.set_fig_size(430, fraction=1, subplots=(1, 1))
    )
    ax = sns.heatmap(
        cmat[::-1],
        annot=True,
        fmt="d",
        linewidths=0.5,
        square=True,
        cbar=False,
        cmap=plt.cm.get_cmap("Blues", np.max(cmat)),
        linecolor="k",
        ax=axis,
    )
    ax.set_ylim(0, n_classes)

    ax.set_ylabel("Ground truth", weight="bold")
    ax.set_yticklabels(np.arange(1, n_classes + 1), ha="right", va="center", rotation=0)

    ax.set_title("Predicted", weight="bold")
    ax.set_xticklabels(
        np.arange(1, n_classes + 1)[::-1], ha="center", va="bottom", rotation=0
    )
    ax.xaxis.set_ticks_position("top")

    fig.tight_layout()
    fig.savefig(
        path_to_figure / f"confusion_{fname}.pdf", transparent=True, bbox_inches="tight"
    )
    plt.close()


def plot_train_loss(epochs, loss_values, path_to_figure: pathlib.Path, fname=""):
    "PLot the loss values from matrix completion."

    fig = plt.figure(figsize=plot_utils.set_fig_size(430, fraction=1, subplots=(1, 1)))
    axis = fig.gca()

    axis.plot(epochs, loss_values, marker="o", alpha=0.7)

    axis.set_ylabel("Loss", weight="bold")
    axis.set_xlabel("Epoch", weight="bold")

    plot_utils.set_arrowed_spines(fig, axis)

    fig.tight_layout()
    fig.savefig(
        path_to_figure / f"train_loss_{fname}.pdf",
        transparent=True,
        bbox_inches="tight",
    )
    plt.close()


def _micro_roc(x_ohe, p_pred, fpr, tpr, roc_auc):
    # Auxillary function for micro-averaged ROC estimate in multi-class problems

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(x_ohe.ravel(), p_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    return fpr, tpr, roc_auc


def _macro_roc(x_ohe, p_pred, fpr, tpr, roc_auc):
    # Auxillary function for macro-averaged ROC estimates in multi-class problems

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(x_ohe.shape[1])]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(x_ohe.shape[1]):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= x_ohe.shape[1]

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return fpr, tpr, roc_auc


def plot_roc_curve(
    x_true,
    p_pred,
    path_to_figure: pathlib.Path,
    number_of_states=settings.matfact_defaults.number_of_states,
    average="micro",
    fname="",
):
    "Plot a ROC curve"

    x_ohe = label_binarize(x_true, classes=range(1, number_of_states + 1))

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(x_ohe.shape[1]):

        fpr[i], tpr[i], _ = roc_curve(x_ohe[:, i], p_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    if average == "macro":
        fpr, tpr, roc_auc = _macro_roc(x_ohe, p_pred, fpr, tpr, roc_auc)

    elif average == "micro":
        fpr, tpr, roc_auc = _micro_roc(x_ohe, p_pred, fpr, tpr, roc_auc)

    else:
        raise ValueError(f"Invalid average: {average}")

    fig = plt.figure(figsize=plot_utils.set_fig_size(430, fraction=1, subplots=(1, 1)))
    axis = fig.gca()

    axis.plot([0, 1], [0, 1], linestyle="--", lw=1.5, color="gray", label="Random")
    axis.plot(
        fpr[average],
        tpr[average],
        lw=1.5,
        label="ROC curve (AUC = %0.2f)" % roc_auc[average],
    )

    axis.set_xlabel("1 - Specificity")
    axis.set_ylabel("Sensitivity")
    axis.set_aspect("equal")

    axis.legend(
        loc="lower left", bbox_to_anchor=(0.5, 0), ncol=1, fancybox=True, shadow=True
    )

    plot_utils.set_arrowed_spines(fig, axis)

    plt.tight_layout()
    plt.savefig(path_to_figure / f"roc_auc_{average}_{fname}.pdf")
    plt.close()


def _calculate_delta(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    correct_indices: Sequence[int] | np.ndarray,
) -> list[float]:
    """Calculate the delta value from a list of probabilities for different classes.

    Args:
        probabilities: (N x number_of_states) with probabilities for each state.
        correct_indices: (N) the index of the correct state per individual.

    Returns:
        The delta score per sample (individual).
    """
    deltas = []
    for estimates, correct in zip(probabilities, correct_indices):
        incorrect_estimates = (*estimates[:correct], *estimates[correct + 1 :])
        # Set default=0 for the edge case that there is only one state, in which
        # case incorrect_estimates is empty.
        deltas.append(max(incorrect_estimates, default=0) - estimates[correct])
    return deltas


def plot_certainty(
    p_pred: Sequence[Sequence[float]] | np.ndarray,
    x_true: np.ndarray,
    path_to_figure: Optional[pathlib.Path] = None,
):
    """Plot the certainty difference delta.

    p_pred is an (number_of_individuals x number_of_states) ndarray.

    Given some classification prediction, let delta = p_(max not correct) - p_correct,
    i.e. the difference in probability between the most likely state that is not
    correct and the state that is correct. For a perfect classifier, this score is -1,
    as the probabilities of all states that are not correct are 0. The worst score is
    +1, where the classifier is completely certain of the wrong state.

    We want to plot the distribution of delta for the individuals N."""

    correct_index = x_true.astype(int) - 1  # x_true is one-indexed
    deltas = _calculate_delta(p_pred, correct_index)
    # deltas are elements in [-1, 1]
    distribution_plot = sns.displot(deltas, kind="ecdf").set(xlim=(-1, 1))

    if path_to_figure is None:
        distribution_plot.fig.show()
    else:
        distribution_plot.savefig(path_to_figure / "certainty_plot.pdf")
