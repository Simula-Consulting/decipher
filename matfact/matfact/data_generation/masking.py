import numpy as np


def simulate_mask(D, observation_proba, memory_length, level, seed=42):
    """Simulate a missing data mask."""

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
