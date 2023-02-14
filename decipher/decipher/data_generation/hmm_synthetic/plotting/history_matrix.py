import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap

from decipher.data_generation.hmm_synthetic.plotting import plot_config, plot_utils

plot_config.setup()


def mask_missing(history: np.ndarray, missing: int = 0) -> np.ndarray:
    """Mask entries with non-observed values as np.nan.

    TODO: This should probably be removed in the future, as
    it seems like an unnecessary function...

    TODO: This is also a copy of hitory_panel::mask_missing
    """

    history[history == missing] = np.nan
    return history


def sample_histories(histories: np.ndarray, size: int, seed: int) -> np.ndarray:
    """Sample size histories from all histories.

    TODO: this is a special case of history_panel::smaple_histories"""

    np.random.seed(seed)
    idx = np.random.choice(range(histories.shape[0]), size=size)

    return histories[idx]


def set_cbar(hmap, axis):  # type: ignore
    """Add and configure colorbar.

    Assumes that there is a current activated figure.
    """

    cbar = plt.colorbar(hmap.get_children()[0], ax=axis, shrink=0.5)
    cbar.set_ticks(np.linspace(1, 4, 4))
    cbar.set_ticklabels(np.array(["Normal", "Low grade", "High grade", "Cancer"]))
    cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), va="center", ha="right")


def plot_history_matrix(
    histories: np.ndarray,
    path_to_figure: pathlib.Path,
    n_samples: int = 500,
    alpha: float = 0.6,
    fname: str = "",
) -> None:
    """Crate an heatplot of all the generated histories."""

    cmap = ListedColormap(
        [
            (0, 0.8, 0, alpha),
            (0.8, 0.8, 0, alpha),
            (0.8, 0.4, 0, alpha),
            (0.8, 0, 0, alpha),
        ]
    )
    cmap.set_under("white", alpha=alpha)

    fig = plt.figure(
        figsize=plot_utils.set_fig_size(430, fraction=1.0, subplots=(1, 1))
    )
    axis: plt.Axes = fig.gca()  # type: ignore

    hmap = sns.heatmap(
        histories[:n_samples],
        ax=axis,
        vmin=0.5,
        vmax=4.5,
        cmap=cmap,
        cbar=False,
        xticklabels=False,  # type: ignore
        yticklabels=False,  # type: ignore
    )

    axis.set_xlabel("Age")
    axis.set_ylabel("History")

    axis.axhline(y=0, color="k", linewidth=1)  # type: ignore
    axis.axhline(y=n_samples, color="k", linewidth=1)  # type: ignore
    axis.axvline(x=0, color="k", linewidth=1)  # type: ignore
    axis.axvline(x=histories.shape[1], color="k", linewidth=1)  # type: ignore

    # ax.tick_params(axis='both', labelbottom=False, labelleft=False)

    set_cbar(hmap, axis)

    fig.tight_layout()
    fig.savefig(
        path_to_figure / f"history_matrix_{fname}.pdf",
        transparent=True,
        bbox_inches="tight",
    )
