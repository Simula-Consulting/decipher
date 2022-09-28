import numpy as np

from backend.utils import age_partitions_pts
from backend.sojourn import time_exit_state
from backend.transition import next_state, inital_state


def simulate_state_histories(n_samples, points_per_year, seed=42):
    """Create a matrix of simulated state histories
    
    Args:
        n_samples: Number of histories 
        seed: Reference value for pseudo-random generator 

    Returns:
        A matrix of simulated state histories.
    
    """

    rnd = np.random.RandomState(seed=seed)

    # Time grid defined by temporal resolution 
    time_grid = age_partitions_pts(points_per_year)

    histories = []
    age_min_pts, age_max_pts = 0, np.max(time_grid)

    for _ in range(n_samples):
        histories.append(_simulate_history(age_min_pts, age_max_pts, time_grid, rnd=rnd))

    return np.array(histories)


def _simulate_history(age_min_pts: int, age_max_pts: int, time_grid: np.ndarray, censoring: int = 0, rnd=None) -> np.ndarray:
    
    history = np.ones(np.max(time_grid)) * censoring

    # Sample the initial state 
    current_state = inital_state(init_age_pts=age_min_pts, time_grid=time_grid, rnd=rnd)

    # Iterate until censoting age 
    current_age_pts = age_min_pts
    while current_age_pts < age_max_pts:

        # Time spent in current state.
        next_age_pts = int(round(time_exit_state(current_age_pts, age_max_pts, current_state, time_grid, rnd=rnd)))

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
    