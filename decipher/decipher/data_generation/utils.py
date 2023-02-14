import numpy as np
from scipy.stats import betabinom

def censoring(X, a, b, missing=0):
    "Truncate histories to have patterns similar to the real histories"

    t_cens = betabinom.rvs(
        n=X.shape[1], a=a, b=b, size=X.shape[0]
    )
    for i, t_end in enumerate(t_cens):
        X[i, t_end:] = missing

    return X


def simulate_mask(D, observation_proba, memory_length, level, seed=42):
    """Simulate a missing data mask."""
    observation_proba = np.array(observation_proba)
    np.random.seed(seed)
    N, T = np.shape(D)

    mask = np.zeros_like(D, dtype=bool)
    observed_values = np.zeros_like(D, dtype=np.float32)

    for t in range(T - 1):

        # Find last remembered values
        observed_cols = (t + 1) - np.argmax(
            observed_values[:, t + 1 : max(0, t - memory_length) : -1] != 0, axis=1
        )
        last_remembered_values = observed_values[np.arange(N), observed_cols]

        p = level * observation_proba[(last_remembered_values).astype(int)]
        r = np.random.uniform(size=N)
        mask[r <= p, t + 1] = True
        observed_values[r <= p, t + 1] = D[r <= p, t + 1]

    return mask


def time_point_approx(T):
    # Approximate number of points per year to
    # obtain T time points with the HMM simulation
    # Will default to 6 points if 0 => T > 400
    # Set the number of data points per year (i.e., temporal resolution)
    # Example: 4 points per year => time between data points is 12 / 4 = 3 months
    time_points = 6
    if 300 <= T < 400:
        time_points = 4
    elif 200 <= T < 300:
        time_points = 3
    elif 100 <= T < 200:
        time_points = 2
    elif 0 <= T < 100:
        time_points = 1
    return time_points


def check_matrix(m, N, T, r):
    # Check matrix m has appropriate dimensions
    N_m, T_m = m.shape
    if N_m != N:
        N = N_m  # Warn?
    if T_m != T:
        T = T_m  # Warn?
    # Check matrix dimensions fullfill requirements
    if N < 1 or T < 1:
        raise ValueError("N and T must be larger than zero.")
    if r > min(N, T):
        raise ValueError("Rank r cannot be larger than either N or T.")
    return N, T