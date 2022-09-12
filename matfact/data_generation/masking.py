import numpy as np


def random_subsample(Y, N, seed=42):

    np.random.seed(seed)
    idx = np.random.choice(range(Y.shape[0]), replace=False, size=N)

    return Y[idx]


def simulate_mask(D, observation_proba, memory_length, level, seed=42):
    """Simulate a missing data mask."""

    np.random.seed(seed)
    N, T = np.shape(D)

    mask = np.zeros_like(D, dtype=np.bool)
    observed_values = np.zeros_like(D, dtype=np.float32)

    for t in range(T - 1):

        # Find last remembered values
        observed_cols = (
            t
            + 1
            - np.argmax(
                observed_values[:, t + 1 : max(0, t - memory_length) : -1] != 0, axis=1
            )
        )
        last_remembered_values = observed_values[np.arange(N), observed_cols]

        p = level * observation_proba[(last_remembered_values).astype(int)]
        r = np.random.uniform(size=N)
        mask[r <= p, t + 1] = True
        observed_values[r <= p, t + 1] = D[r <= p, t + 1]

    return mask


def trim_time_axis(X, dt, t_min=None, t_max=None):

    t_min = t_min if t_min is not None else 16
    t_max = t_max if t_max is not None else 100

    # Dots per year.
    dpy = 12 // dt

    t_min_idx = int((t_min - 16) * dpy)
    t_max_idx = int((t_max - 16) * dpy + 1)

    return X[:, t_min_idx:t_max_idx]


def thresh_observation_count(X, min_n_obs=None, max_n_obs=None, meta=None):
    """Control the number of observations per individual."""

    nz = np.count_nonzero(X, axis=1)

    if min_n_obs is not None and max_n_obs is not None:
        to_keep = np.logical_and(nz >= min_n_obs, nz <= max_n_obs)

    if min_n_obs is not None and max_n_obs is None:
        to_keep = nz >= min_n_obs

    if min_n_obs is None and max_n_obs is not None:
        to_keep = nz <= max_n_obs

    if meta is not None:
        return X[to_keep], meta.iloc[to_keep]

    return X[to_keep]
