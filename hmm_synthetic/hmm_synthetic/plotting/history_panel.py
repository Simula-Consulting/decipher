import matplotlib.pyplot as plt
import numpy as np

from . import plot_config, plot_utils

plot_config.setup()


def mask_missing(history, missing=0):

    history[history == missing] = np.nan
    return history


def sample_histories(histories, size, rnd, risk_stratify=False):

    s_max = np.ones(histories.shape[0])

    if risk_stratify:
        s_max = np.max(histories, axis=1)

    idx = rnd.choice(range(histories.shape[0]), size=size, p=s_max / sum(s_max))

    return histories[idx]


def plot_history_panel(
    histories,
    path_to_figure,
    points_per_year,
    age_min=16,
    age_max=100,
    fname="",
    rnd=None,
):
    """Plot a panel of state histories."""

    fig, axes = plt.subplots(
        3, 2, figsize=plot_utils.set_fig_size(430, fraction=1, subplots=(3, 2))
    )

    histories = sample_histories(histories, 6, rnd)

    for i, axis in enumerate(axes.ravel()):

        history = histories[i]
        time_grid = np.linspace(0, histories.shape[1], 6)

        axis.plot(mask_missing(history), marker="o", linestyle="")

        axis.set_ylabel("States")
        axis.set_yticks([1, 2, 3, 4])
        axis.set_yticklabels([1, 2, 3, 4])

        axis.set_ylim([0.5, 4.5])

        axis.set_xlabel("Age")
        axis.set_xticks(time_grid)
        axis.set_xticklabels(np.round(time_grid / points_per_year + age_min, 2))

        plot_utils.set_arrowed_spines(fig, axis)

    fig.tight_layout()
    fig.savefig(
        path_to_figure / f"history_panel_{fname}.pdf",
        transparent=True,
        bbox_inches="tight",
    )
