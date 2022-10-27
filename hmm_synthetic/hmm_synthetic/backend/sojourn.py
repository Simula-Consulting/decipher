from typing import Sequence

import numpy as np
import numpy.typing as npt

from .transition import legal_transition_lambdas
from .utils import age_group_idx


def kappa_0(age, current_state, t, time_grid) -> float:

    l = age_group_idx(age + t, time_grid)  # noqa: E741

    return -1.0 * t * sum(legal_transition_lambdas(current_state, l))


def kappa_1(age, current_state, t, time_grid, l=None) -> float:  # noqa: E741

    k = age_group_idx(age, time_grid)
    tau_kp = time_grid[k + 1]

    if l is None:
        l = age_group_idx(age + t, time_grid)  # noqa: E741

    tau_l = time_grid[l]

    s_k = (tau_kp - age) * sum(legal_transition_lambdas(current_state, k))
    s_l = (age - tau_l) * sum(legal_transition_lambdas(current_state, l))

    return (-1.0 * s_k) - s_l


def kappa_m(age, current_state, m, time_grid) -> float:

    k = age_group_idx(age, time_grid)

    tau_km = time_grid[k + m - 1]
    tau_kmp = time_grid[k + m]

    s_km = (tau_kmp - tau_km) * sum(legal_transition_lambdas(current_state, k + m - 1))

    return -1.0 * s_km


def kappa(age, current_state, t, i, time_grid, l=None) -> float:  # noqa: E741

    if i == 0:
        return kappa_0(age, current_state, t, time_grid)

    if i == 1:
        return kappa_1(age, current_state, t, time_grid, l=l)

    return kappa_m(age, current_state, i, time_grid)


def search_l(u, k, current_age_pts, state, time_grid, n_age_partitions=8):

    if np.isclose(k, n_age_partitions - 1):
        return k

    for l in range(k, n_age_partitions):  # noqa: E741

        tau_lp = time_grid[l + 1]

        t = tau_lp - current_age_pts

        # Evaluate the sojourn time CDF at this time step.
        cdf = 1.0 - np.exp(
            sum(
                [
                    kappa(current_age_pts, state, t, i, time_grid)
                    for i in range(l - k + 1)
                ]
            )
        )

        if cdf > u:
            return l - 1

    return l - 1


def exit_time(u, a, s, k, l, time_grid):  # noqa: E741
    """Random exit time from current state."""

    sum_kappa = sum([kappa(a, s, None, i, time_grid, l=l) for i in range(1, l - k + 1)])

    return (sum_kappa - np.log(1 - u)) / sum(legal_transition_lambdas(s, l))


def time_exit_state(
    current_age_pts: int,
    age_max_pts: int,
    state: int,
    time_grid: Sequence[int] | npt.NDArray[np.int_],
    rnd=None,
) -> float:
    """Returns the amount of time a female spends in the current state."""

    # Need t > 0.
    # TODO: Is this line correct, or does it assume ints?
    # exit_time can output a number x s.t. 0 < x < 1.
    # Should the test instead be <= 0?
    if abs(age_max_pts - current_age_pts) <= 1:
        return age_max_pts

    # Corollary 1: Steps 1-4
    u = rnd.uniform()

    k = age_group_idx(current_age_pts, time_grid)

    l = search_l(u, k, current_age_pts, state, time_grid)  # noqa: E741

    return exit_time(u, current_age_pts, state, k, l, time_grid)
