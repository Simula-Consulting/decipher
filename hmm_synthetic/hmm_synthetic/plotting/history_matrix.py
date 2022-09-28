import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 

from matplotlib.colors import ListedColormap

from . import plot_config
from . import plot_utils 

plot_config.setup()


def mask_missing(history, missing=0):

    history[history == missing] = np.nan 
    return history


def sample_histories(histories, size, seed):

    np.random.seed(seed)
    idx = np.random.choice(range(histories.shape[0]), size=size)

    return histories[idx]


def set_cbar(hmap, axis):

    cbar = plt.colorbar(hmap.get_children()[0], ax=axis, shrink=0.5)
    cbar.set_ticks(np.linspace(1, 4, 4))
    cbar.set_ticklabels(np.array(["Normal", "Low grade", "High grade", "Cancer"]))
    cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), va='center', ha="right") 


def plot_history_matrix(histories, path_to_figure, n_samples=500, alpha=0.6, fname=""):

    cmap = ListedColormap([(0, 0.8, 0, alpha), (0.8, 0.8, 0, alpha), 
                           (0.8, 0.4, 0, alpha), (0.8, 0, 0, alpha)])
    cmap.set_under('white', alpha=alpha)

    fig, axis = plt.subplots(1, 1, figsize=plot_utils.set_fig_size(430, fraction=1.0, subplots=(1, 1)))

    hmap = sns.heatmap(histories[:n_samples], ax=axis, vmin=0.5, vmax=4.5, 
                        cmap=cmap, cbar=False, xticklabels=False, yticklabels=False)

    axis.set_xlabel("Age")
    axis.set_ylabel("History")

    axis.axhline(y=0, color='k', linewidth=1)
    axis.axhline(y=n_samples, color='k', linewidth=1)
    axis.axvline(x=0, color='k', linewidth=1)
    axis.axvline(x=histories.shape[1], color='k', linewidth=1)

    #ax.tick_params(axis='both', labelbottom=False, labelleft=False)

    set_cbar(hmap, axis)

    fig.tight_layout()
    fig.savefig(f"{path_to_figure}/history_matrix_{fname}.pdf", transparent=True, bbox_inches="tight")