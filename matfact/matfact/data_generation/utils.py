import numpy as np


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


def matrix_info(m: np.ndarray) -> list:
    # Round values and return value count list within the given matrix
    return [
        (int(value), int(count))
        for value, count in np.vstack(np.unique(np.round(m), return_counts=True)).T
    ]


def invert_matrix_domain(m: np.ndarray) -> np.ndarray:
    # Inverts matrix label distributions
    return m.max() - (m - m.min())


def invert_matrix_domain_excluding_value(m: np.ndarray, exclude: int) -> np.ndarray:
    # Inverts matrix label distributions excluding a value
    inv_m = np.zeros(m.shape)
    inv_m[m != exclude] = invert_matrix_domain(m[m != exclude])
    return inv_m


def invert_observation_probabilities(obs_probabilities: list) -> list:
    # Inverts list of observation probabilities,
    # leaving the missing data (0) unchanged.
    return [obs_probabilities[0]] + obs_probabilities[-1:0:-1]
