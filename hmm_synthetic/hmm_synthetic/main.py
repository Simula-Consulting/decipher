import numpy as np

from hmm_synthetic.data_generator import simulate_state_histories
from hmm_synthetic.plotting.history_matrix import plot_history_matrix
from hmm_synthetic.plotting.history_panel import plot_history_panel
from hmm_synthetic.settings import settings


def main():

    # Set the number of data points per year (i.e., temporal resolution)
    # Example: 4 points per year => time between data points is 12 / 4 = 3 months
    points_per_year = 4

    # Simulate fixed length state histories
    D = simulate_state_histories(
        n_samples=100, points_per_year=points_per_year, seed=42
    )

    plot_history_matrix(D, path_to_figure=settings.paths.figure, fname="hmm")

    # Use different random seeds to sample histories for plotting
    rnd = np.random.RandomState(seed=42)
    seeds = rnd.choice(range(1000), size=5, replace=False)

    for i, seed in enumerate(seeds):
        plot_history_panel(
            D,
            path_to_figure=settings.paths.figure,
            points_per_year=points_per_year,
            rnd=rnd,
            fname=f"D{i}",
        )


if __name__ == "__main__":
    main()
