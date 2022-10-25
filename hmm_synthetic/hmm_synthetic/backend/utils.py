from typing import Sequence

import numpy as np
import numpy.typing as npt
import scipy.stats as stats

age_partitions = np.array([16, 20, 25, 30, 35, 40, 50, 60, 100])

# Converting to pts.
# age_partitions_pts = np.round(
#     (age_partitions - np.min(age_partitions)) * 4
# ).astype(int)


def age_partitions_pts(points_per_year: int) -> npt.NDArray[np.int_]:
    """Convert age partitions from year to time discretization."""

    #   a = floor(pts / n) + a0    <==>    pts = (a - a0) * n
    return np.round((age_partitions - np.min(age_partitions)) * points_per_year).astype(
        int
    )


# Initial state probabilities: age group x probability initial state (adjusted to row
# stochastic).
p_init_state = np.array(
    [
        [0.92992, 0.06703, 0.00283, 0.00022],
        [0.92881, 0.06272, 0.00834, 0.00013],
        [0.93408, 0.04945, 0.01632, 0.00015],
        [0.94918, 0.03554, 0.01506, 0.00022],
        [0.95263, 0.03250, 0.01463, 0.00024],
        [0.95551, 0.03303, 0.01131, 0.00015],
        [0.96314, 0.02797, 0.00860, 0.00029],
        [0.96047, 0.02747, 0.01168, 0.00038],
    ]
)

# Transition intensities: age group x state transition.
# fmt: off
lambda_sr = np.array(
    [
        # N0->L1  L1->H2   H2->C3   L1->N0   H2->L1   N0->D4   L1->D4   H2->D4   C3->D4
        [0.02027, 0.01858, 0.00016, 0.17558, 0.24261, 0.00002, 0.00014, 0.00222, 0.01817],  # noqa: E501
        [0.01202, 0.02565, 0.00015, 0.15543, 0.11439, 0.00006, 0.00016, 0.00082, 0.03024],  # noqa: E501
        [0.00746, 0.03959, 0.00015, 0.14622, 0.07753, 0.00012, 0.00019, 0.00163, 0.03464],  # noqa: E501
        [0.00584, 0.04299, 0.00055, 0.15576, 0.07372, 0.00012, 0.00016, 0.00273, 0.04211],  # noqa: E501
        [0.00547, 0.03645, 0.00074, 0.15805, 0.06958, 0.00010, 0.00015, 0.00398, 0.04110],  # noqa: E501
        [0.00556, 0.02970, 0.00127, 0.17165, 0.08370, 0.00010, 0.00014, 0.00518, 0.03170],  # noqa: E501
        [0.00440, 0.02713, 0.00161, 0.19910, 0.10237, 0.00020, 0.00029, 0.00618, 0.02772],  # noqa: E501
        [0.00403, 0.03826, 0.00419, 0.24198, 0.06951, 0.00104, 0.00115, 0.02124, 0.02386],  # noqa: E501
    ]
)
# fmt: on


def age_group_idx(
    a: int, age_partitions_pts: Sequence[tuple[int, int]] | npt.NDArray[np.int_]
) -> int:
    """Returns index i: tau_i <= a < tau_{i+1}."""
    if a < age_partitions_pts[0][0]:
        raise ValueError("Age is smaller than the first age partition!")
    if a > age_partitions_pts[-1][-1]:
        raise ValueError("Age is higher than the last age partition!")

    for i, (tau_p, tau_pp) in enumerate(age_partitions_pts):

        if np.logical_and(tau_p <= a, a < tau_pp):
            return i

    # If a = tau_pp at the last age group
    return i


def _start_end_times(N, time_grid, params, rnd=None):
    """Sample times for init and final screenings."""

    time_grid = np.arange(T)  # noqa: F821

    # Probability distributions for the first and last observed state
    # p_start = stats.exponnorm.pdf(x=time_grid, K=8.76, loc=9.80, scale=7.07)
    # p_cens = stats.exponweib.pdf(
    #     x=time_grid, a=513.28, c=4.02, loc=-992.87, scale=707.63,
    # )

    p_start = stats.lognomr(
        time_grid,
        s=params["lnorm_s"],
        loc=params["lnorm_loc"],
        scale=params["lnorm_scale"],
    )
    p_end = stats.uniform(  # noqa: F841
        time_grid, loc=params["uni_loc"], scale=params["uni_scale"]
    )
    # p_end = stats.norm(time_grid, loc=params["norm_loc"], scale=params["norm_scale"])

    if rnd is None:
        t_start = np.random.choice(time_grid, size=N, p=p_start / sum(p_start))
    else:
        t_start = rnd.choice(time_grid, size=N, p=p_start / sum(p_start))

    if rnd is None:
        t_cens = np.random.choice(
            time_grid, size=N, p=p_cens / sum(p_cens)  # noqa: F821
        )
    else:
        t_cens = rnd.choice(time_grid, size=N, p=p_cens / sum(p_cens))  # noqa: F821

    # Ensure t_end > t_start for all histories
    to_keep = np.squeeze(np.argwhere(t_cens - t_start > 10))

    return t_start[to_keep], t_cens[to_keep], to_keep
