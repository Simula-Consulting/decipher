from typing import Sequence

import numpy as np
import numpy.typing as npt
import scipy.stats as stats

from decipher.data_generation.settings import hmm_settings as settings

# Converting to pts.
# age_partitions_pts = np.round(
#     (age_partitions - np.min(age_partitions)) * 4
# ).astype(int)


def age_partitions_pts(points_per_year: int) -> npt.NDArray[np.int_]:
    """Convert age partitions from year to time discretization."""
    age_partitions = settings.static.age_partitions
    #   a = floor(pts / n) + a0    <==>    pts = (a - a0) * n
    return np.round((age_partitions - np.min(age_partitions)) * points_per_year).astype(
        int
    )


def age_group_idx(
    age: int, age_partitions_pts: Sequence[int] | npt.NDArray[np.int_]
) -> int:
    """Returns index i: tau_i <= age < tau_{i+1}.

    TODO: What is the correct behavior in the case that age == max(age_partitions)?
    Then, no index fulfills the criterion age < tau_{i+1}.
    As of now, we have chosen to take the last partition to be inclusive, i.e.
    age <= tau_{max}, as this was the case in the original code, and we have reason
    to believe that our dataset will contain ages including the endpoint of the
    oldest partition.
    """
    if age < age_partitions_pts[0]:
        raise ValueError("Age is smaller than the first age partition!")
    if age > age_partitions_pts[-1]:
        raise ValueError("Age is higher than the last age partition!")
    if len(age_partitions_pts) <= 1:
        raise ValueError(
            "age_partitions_pts must have at least two elements to define a partition!"
        )

    # Break at the first partition where the age is smaller than the upper limit
    # The last partition is different from the rest, as it has an inclusive upper limit,
    # as opposed to the rest which have exclusive lower limit.
    # Due to this, we do not return inside the loop but break.
    # For the last partition, the if statement will thus never be True, but it will
    # nevertheless reach the return statement.
    for i, partition_upper_limit in enumerate(age_partitions_pts[1:]):
        if age < partition_upper_limit:
            break
    return i  # pylint: disable=undefined-loop-variable  # Guard at beginning assures iterator is not empty


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
