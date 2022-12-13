from typing import Sequence

import numpy as np
import numpy.typing as npt
import scipy
import tensorflow as tf
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


class Projection(Constraint):
    """Constraint class implementing both constraints on the optimization variable.

    The two constraints are:
       1. Element norm: Individual entries of A are bounded by some value gamma:
          |A|_max <= gamma
       2. Nuclear norm: The nuclear norm is bounded by some value tau:
          |A|_* <= tau * sqrt(m*n)

    Tau and gamma are user specified.
    """

    def __init__(self, tau_mn: float, gamma: float):
        self.tau_mn = tau_mn
        self.gamma = gamma

    def element_norm(self, _A: tf.Tensor) -> tf.Tensor:
        return tf.clip_by_value(
            _A, clip_value_max=self.gamma, clip_value_min=-self.gamma
        )

    def _simplex_projection(
        self, s: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """Projection onto the unit simplex."""
        if np.sum(s) <= self.tau_mn and np.alltrue(s >= 0):
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

    def _nuclear_projection(
        self,
        s: npt.NDArray[np.float32],
        U: npt.NDArray[np.float32],
        V: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """Projection onto nuclear norm ball."""
        s = self._simplex_projection(s)
        return U.dot(np.diag(s).dot(V))

    def nuclear_norm(self, _A: tf.Tensor) -> npt.NDArray[np.float32]:
        U, s, V = scipy.linalg.svd(_A, full_matrices=False)
        return self._nuclear_projection(s, U, V)

    def __call__(self, A: tf.Tensor) -> tf.Tensor:
        return self.element_norm(self.nuclear_norm(A))


def get_one_zero_mask(
    M: npt.NDArray[np.int_] | tf.Tensor,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    return (M == 1), (M == 0)


def calculate_propensity_scores(
    M: tf.Tensor, learning_rate: float, n_iter: int, tau: float, gamma: float
) -> tf.Tensor:
    """Function to calculate a propensity score matrix based on a binary observation
    matrix. The propensity weights are calculated from the maximum likelihood equation
    (Eq. (3) in Ma & Chen (2019)) using projected gradient descent.
    The projections are included as constraints on the variable 'A'.

    Currently we are only considering using the sigmoid function as a link function as
    this seems to be the general consensus, but it could in theory be any function.
    """
    M_one_mask, M_zero_mask = get_one_zero_mask(M)
    m, n = M.shape
    tau_mn = tau * np.sqrt(m * n)

    constraints = Projection(tau_mn=tau_mn, gamma=gamma)

    A = tf.Variable(tf.random.uniform(M.shape), constraint=constraints)
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    def loss():
        term1 = reduce_sum(log(sigmoid(A[M_one_mask])))
        term2 = reduce_sum(log(1 - sigmoid(A[M_zero_mask])))
        return -(term1 + term2)

    for _ in range(n_iter):
        optimizer.minimize(loss, var_list=[A])

    return sigmoid(A)


def propensity_weights(
    observed_data_matrix: npt.NDArray[np.int_],
    learning_rate: float = 0.1,
    n_iter: int = 100,
    tau: float = settings.DEFAULT_TAU,
    gamma: float = settings.DEFAULT_GAMMA,
) -> npt.NDArray[np.float_]:
    """Construct weight matrix scaled by the inverse propensity score.
    The propensity scores are calculated from a binary map of the observation matrix.
    """
    M = np.zeros_like(observed_data_matrix)
    M[observed_data_matrix != 0] = 1
    M = tf.cast(M, dtype=tf.float32)

    propensity_scores = calculate_propensity_scores(
        M, learning_rate=learning_rate, n_iter=n_iter, tau=tau, gamma=gamma
    )

    return (M / propensity_scores).numpy()
