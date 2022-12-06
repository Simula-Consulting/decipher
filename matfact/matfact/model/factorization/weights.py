from typing import Sequence

import numpy as np
import numpy.typing as npt
import tensorflow as tf
from tensorflow.math import log, sigmoid

from matfact import settings


def data_weights(
    observed_data_matrix: npt.NDArray[np.int_], weights: Sequence[float] | None = None
):
    """Construct a weight matrix for observed data.

    weights contains the weights that should be given to the various states. I.e. the
    first element of weights is the weighing of state 1, the second element for state 2,
    and so on.

    Raises ValueError if there are observed states for which no weight is given.
    """
    if weights is None:
        weights = settings.default_weights

    assert np.min(observed_data_matrix) >= 0  # Observed data should never be negative
    if np.max(observed_data_matrix) > len(weights):
        raise ValueError("The observed data have states for which no weight is given.")
    weight_matrix = np.zeros(observed_data_matrix.shape)
    for i, weight in enumerate(weights):
        state = i + 1  # States are one indexed
        weight_matrix[observed_data_matrix == state] = weight
    return weight_matrix


def propensity_weights(
    observed_data_matrix: npt.NDArray[np.int_], lr=0.1, n_iter=100
) -> npt.NDArray:
    """Construct weight matrix scaled by the inverse propensity score.

    Solves the Bernoulli maximum likelihood problem (Eq. (3) in Ma & Chen (2019))
    by minimizing a loss function defined as the opposite of the maximum likelihood.

    TODO: Implement tau and gamma contraints (if necessary)
    """
    M = np.zeros_like(observed_data_matrix)
    M[observed_data_matrix != 0] = 1
    M = tf.cast(M, dtype=tf.float32)

    A = tf.Variable(tf.random.uniform(M.shape))
    optimizer = tf.optimizers.Adam(learning_rate=lr)

    def loss():
        term1 = (1 - M) * log(1 - sigmoid(A))
        term2 = M * log(sigmoid(A))
        return -tf.linalg.trace(term1 + term2)

    for _ in range(n_iter):
        optimizer.minimize(loss, var_list=[A])

    propensity_matrix = sigmoid(A)
    return (M / propensity_matrix).numpy()
