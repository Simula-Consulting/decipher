from typing import Sequence

import numpy as np
import numpy.typing as npt
import tensorflow as tf
from scipy.linalg import svd
from tensorflow.keras.constraints import Constraint
from tensorflow.math import log, reduce_sum, sigmoid

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
    observed_data_matrix: npt.NDArray[np.int_],
    lr=0.1,
    n_iter=1000,
    tau=None,
    gamma=None,
) -> npt.NDArray:
    """Construct weight matrix scaled by the inverse propensity score.

    Solves the Bernoulli maximum likelihood problem (Eq. (3) in Ma & Chen (2019))
    by minimizing a loss function defined as the opposite of the maximum likelihood.
    """
    M = np.zeros_like(observed_data_matrix)
    M[observed_data_matrix != 0] = 1
    M = tf.cast(M, dtype=tf.float32)
    M_one_mask = M == 1
    M_zero_mask = M == 0

    m, n = M.shape
    tau_mn = (tau or settings.DEFAULT_TAU) * np.sqrt(m * n)
    gamma = settings.DEFAULT_GAMMA

    constraints = BothConstraints(tau_mn=tau_mn, gamma=gamma)

    A = tf.Variable(tf.random.uniform(M.shape), constraint=constraints)
    optimizer = tf.optimizers.Adam(learning_rate=lr)

    def loss():
        term1 = tf.reduce_sum(log(sigmoid(A.numpy()[M_one_mask])))
        term2 = reduce_sum(log(1 - sigmoid(A[M_zero_mask])))
        return -(term1 + term2)

    for _ in range(n_iter):
        optimizer.minimize(loss, var_list=[A])

    propensity_matrix = sigmoid(A)
    return (M / propensity_matrix).numpy()


class BothConstraints(Constraint):
    def __init__(self, tau_mn, gamma):
        self.tau_mn = tau_mn
        self.gamma = gamma

    def element_norm(self, _A):
        return tf.clip_by_value(
            _A, clip_value_max=self.gamma, clip_value_min=-self.gamma
        )

    def _simplex_projection(self, s):
        """Projection onto the unit simplex."""
        if np.sum(s) <= self.tau_mn and np.alltrue(s >= 0):
            # no need to project!
            return s
        # get the array of cumulative sums of a sorted (decreasing) copy of s
        u = np.sort(s)[::-1]
        cssv = np.cumsum(u)

        # get the number of > 0 components of the optimal solution
        rho = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - self.tau_mn))[0][-1]
        # compute the Lagrange multiplier associated to the simplex constraint
        theta = (cssv[rho] - self.tau_mn) / (rho + 1)
        # compute the projection by thresholding s using theta
        return np.maximum(s - theta, 0)

    def _nuclear_projection(self, s, U, V):
        """Projection onto nuclear norm ball."""
        s = self._simplex_projection(s)
        return U.dot(np.diag(s).dot(V))

    def nuclear_norm(self, _A):
        U, s, V = svd(_A, full_matrices=False)
        return self._nuclear_projection(s, U, V)

    def __call__(self, A):
        A = A.numpy()
        A = self.nuclear_norm(A)
        return self.element_norm(A)
