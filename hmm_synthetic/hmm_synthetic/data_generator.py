import numpy as np

from .backend.sojourn import time_exit_state
from .backend.transition import initial_state, next_state
from .backend.utils import age_partitions_pts


def simulate_state_histories(
    n_samples: int, points_per_year: int, seed: int = 42
) -> np.ndarray:
    """Create a matrix of simulated state histories

    Args:
        n_samples: Number of histories
        seed: Reference value for pseudo-random generator

    Returns:
        A matrix of simulated state histories.

    """

    rnd = np.random.default_rng(seed=seed)

    # Time grid defined by temporal resolution
    time_grid = age_partitions_pts(points_per_year)

    histories = []
    age_min_pts = 0
    age_max_pts = np.max(time_grid).astype(int)  # Convert from np.int_

    for _ in range(n_samples):
        histories.append(
            _simulate_history(age_min_pts, age_max_pts, time_grid, rnd=rnd)
        )

    return np.array(histories)


def _simulate_history(
    age_min_pts: int,
    age_max_pts: int,
    time_grid: np.ndarray,
    rnd: np.random.Generator,
    censoring: int = 0,
) -> np.ndarray:
    """Simulate history for one individual.

    Args:
        age_min_pts: the first point of the time discretization.
        age_max_pts: the last point of the time discretization.
        time_grid: the time discretized age partitions.
        cencoring: the value of non-observed entries.
        rnd: np.random.RandomState instance
    """
    if age_max_pts <= age_min_pts:
        raise ValueError("The minimum age must be smaller than the maximum age.")
    if age_min_pts < 0:
        raise ValueError("The minimum age cannot be negative.")
    max_age = np.max(time_grid)
    if age_max_pts > max_age:
        raise ValueError("The maximum age cannot exceed the time partitions.")

    history = np.ones(max_age) * censoring

    # Sample the initial state
    current_state = initial_state(
        init_age_pts=age_min_pts, time_grid=time_grid, rnd=rnd
    )

    # Iterate until censoting age
    current_age_pts = age_min_pts
    while current_age_pts < age_max_pts:

        # Time spent in current state.
        next_age_pts = int(
            round(
                time_exit_state(
                    current_age_pts, age_max_pts, current_state, time_grid, rnd=rnd
                )
            )
        )

        if next_age_pts == 0:
            continue

        # Sanity check.
        if next_age_pts < 0:
            raise ValueError(f"Negative age increment: {next_age_pts}")

        # Age when exiting current state.
        next_age_pts += current_age_pts

        # Clip to censoring time. Will break the iterations.
        if next_age_pts >= age_max_pts:
            next_age_pts = age_max_pts

        history[current_age_pts:next_age_pts] = current_state

        # Sample most likely next state from lambdas over legal transitions.
        current_state = next_state(next_age_pts, current_state, time_grid, rnd=rnd)

        # Censoring rest of state vector.
        if current_state == censoring:
            return history

        current_age_pts = next_age_pts

    return history
