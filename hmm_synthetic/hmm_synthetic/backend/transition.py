import numpy as np

from .utils import age_group_idx, lambda_sr, p_init_state


def inital_state(init_age_pts: int, time_grid, rnd=None) -> int:
    """Sample a state at first screening."""

    if rnd is None:
        return np.random.choice([1, 2, 3, 4], p=p_init_state[age_group_idx(init_age_pts, time_grid)])       

    return rnd.choice([1, 2, 3, 4], p=p_init_state[age_group_idx(init_age_pts, time_grid)])       


def legal_transition_lambdas(current_state: int, time_idx: int) -> np.ndarray:
    """Filter intensities for shifts from the current state."""

    # Transition intensities for the given age group.
    lambdas = np.squeeze(lambda_sr[time_idx])

    # N0 -> L1/D4.
    if current_state == 1:
        return np.array([lambdas[0], lambdas[5]])
    
    # L1 -> N0/H2/D4.
    if current_state == 2:
        return np.array([lambdas[3], lambdas[1], lambdas[6]])

    # H2 -> C3/D4/L1
    if current_state == 3:
        return np.array([lambdas[2], lambdas[7], lambdas[4]])

    # C3 -> D4
    if current_state == 4:
        return np.asarray([lambdas[8]])

    raise ValueError(f"Invalid current state: {current_state}")


def next_state(age_at_exit_pts: int, current_state: int, time_grid: np.array, censoring: int = 0, rnd=None) -> int:
    """Returns next female state."""

    # NB: Truncates each history at cancer diagnosis.
    if current_state == 4:
        return censoring

    lambdas = legal_transition_lambdas(current_state, age_group_idx(age_at_exit_pts, time_grid))

    # N0 -> L1/D4 (censoring)
    if current_state == 1:
        return rnd.choice([2, censoring], p=lambdas / sum(lambdas))

    # L1 -> N0/H2/D4
    if current_state == 2:
        return rnd.choice([1, 3, censoring], p=lambdas / sum(lambdas))
    
    # H2 -> C3/D4/L1
    if current_state == 3:
        return rnd.choice([4, censoring, 2], p=lambdas / sum(lambdas))
    
    raise ValueError(f"Invalid current state: {current_state}")