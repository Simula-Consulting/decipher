import pathlib
from typing import cast

import matplotlib.pyplot as plt
import numpy as np

from . import plot_config, plot_utils

plot_config.setup()


def mask_missing(history: np.ndarray, missing: int = 0) -> np.ndarray:
    """Mask entries with non-observed values as np.nan.

    TODO: This should probably be removed in the future, as
    it seems like an unnecessary function..."""

    history[history == missing] = np.nan
    return history


def sample_histories(
    histories: np.ndarray,
    size: int,
    rnd: np.random.Generator,
    risk_stratify: bool = False,
):
    """Sample size histories from all histories.

    Args:
        histories: (number_of_individuals x time_points) array of all histories
        size: number of histories to sample
        rnd: instance of a random generator
        risk_stragtify; if True, samples with higher maximum risk are more
            likely to be sampled

    Returns:
        The sampled histories, (size x time_points).
    """

    s_max = np.ones(histories.shape[0])

    if risk_stratify:
        s_max = np.max(histories, axis=1)

    idx = rnd.choice(range(histories.shape[0]), size=size, p=s_max / sum(s_max))

    return histories[idx]


def plot_history_panel(
    histories: np.ndarray,
    path_to_figure: pathlib.Path,
    points_per_year: int,
    age_min: int = 16,
    age_max: int = 100,
    fname: str = "",
    rnd: np.random.Generator | None = None,
):
    """Plot a panel of state histories."""

    if rnd is None:
        rnd = np.random.default_rng()

    fig, axes_ = plt.subplots(
        3, 2, figsize=plot_utils.set_fig_size(430, fraction=1, subplots=(3, 2))
    )

    # The type hinting of plt.subplots is wrong. The axes return type is
    # typed to be List[List[plt.Axes]], however it is actually a numpy array.
    # We therefore cast manually here.
    #
    # This workaround should be removed as soon as the stubs are corrected.
    #
    # Versions:
    #  - matplotlib: 3.6.0
    #  - data-science-types: 0.2.23
    axes = cast(np.ndarray, axes_)

    histories = sample_histories(histories, 6, rnd)

    axis: plt.Axes
    for i, axis in enumerate(axes.ravel()):
        history = histories[i]
        time_grid = np.linspace(0, histories.shape[1], 6)

        axis.plot(mask_missing(history), marker="o", linestyle="")  # type: ignore

        axis.set_ylabel("States")
        axis.set_yticks([1, 2, 3, 4])
        axis.set_yticklabels([1, 2, 3, 4])  # type: ignore

        axis.set_ylim(0.5, 4.5)

        axis.set_xlabel("Age")
        axis.set_xticks(time_grid)
        axis.set_xticklabels(np.round(time_grid / points_per_year + age_min, 2))  # type: ignore # noqa: E501

        plot_utils.set_arrowed_spines(fig, axis)

    fig.tight_layout()
    fig.savefig(
        path_to_figure / f"history_panel_{fname}.pdf",
        transparent=True,
        bbox_inches="tight",
    )
