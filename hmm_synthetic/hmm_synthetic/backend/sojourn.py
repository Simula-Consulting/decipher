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


def kappa(
    age: int,
    current_state: int,
    i: int,
    time_grid: Sequence[int] | npt.NDArray[np.int_],
    t: int | None = None,
    l_partition_index: int | None = None,
) -> float:

    assert (
        t is not None or l_partition_index is not None
    ), "either t or l_partition_index must be specified"
    if i == 0:
        return kappa_0(age, current_state, t, time_grid)

    if i == 1:
        return kappa_1(age, current_state, t, time_grid, l=l_partition_index)

    return kappa_m(age, current_state, i, time_grid)


def search_l(
    u_random_variable: float,
    age_partition_index: int,
    current_age_pts: int,
    state: int,
    time_grid: Sequence[int] | npt.NDArray[np.int_],
    n_age_partitions: int = 8,
) -> int:
    r"""Find the partition satisfying the probability requirement.

    Find l such that

    $$
    P[T_s(a) < tau_l - age] < u < P[T_s(a) < tau_{l+1} - q],
    $$

    where

    $$
    P[T_s(a) > t] = exp( kappa_0^s + kappa_1^s + sum_{i=2}^n kappa_i^s )
                  = exp( sum_i kappa_i^s)
    $$

    In other words, find the largest l such that
    \[
    P[T_s(a) < tau_l - a] < u.
    \]
    """

    if age_partition_index == n_age_partitions - 1:
        return age_partition_index

    for l_partition_index_candidate in range(age_partition_index, n_age_partitions):

        partition_upper_limit = time_grid[l_partition_index_candidate + 1]

        t = partition_upper_limit - current_age_pts

        # Evaluate the sojourn time CDF at this time step.
        cdf = 1.0 - np.exp(
            sum(
                [
                    kappa(current_age_pts, state, i, time_grid, t=t)
                    for i in range(
                        l_partition_index_candidate - age_partition_index + 1
                    )
                ]
            )
        )

        if cdf > u_random_variable:
            return l_partition_index_candidate - 1

    return l_partition_index_candidate - 1


def exit_time(
    u_random_variable: float,
    age: int,
    state: int,
    age_partition_index: int,
    l_partition_index: int,
    time_grid: Sequence[int] | npt.NDArray[np.int_],
) -> float:
    """Random exit time from current state.

    Notation used
    - age_partition_index: corresponds to k in the paper
    - l_partition_index: corresponds to l in the paper.
        It is the partition index of the exit age (ish)
    """

    sum_kappa = sum(
        [
            kappa(age, state, i, time_grid, l_partition_index=l_partition_index)
            for i in range(1, l_partition_index - age_partition_index + 1)
        ]
    )

    return (sum_kappa - np.log(1 - u_random_variable)) / sum(
        legal_transition_lambdas(state, l_partition_index)
    )


def time_exit_state(
    current_age_pts: int,
    age_max_pts: int,
    state: int,
    time_grid: Sequence[int] | npt.NDArray[np.int_],
    rnd: np.random.Generator,
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
