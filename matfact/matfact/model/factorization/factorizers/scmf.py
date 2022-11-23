from typing import cast

import numpy as np
import tensorflow as tf
from numpy.lib.stride_tricks import as_strided

from matfact.model.config import ModelConfig

from .mfbase import BaseMF


def _custom_roll(arr, m):
    """Roll array elements with different amount per axis.

    Fast implementation of row wise shifting.

    Arguments:
    arr: two-dimensional array
    m: one dimensional list of integers, each corresponding to a shift of a row in arr

    NB! For very large shifts, floating point errors may cause the wrong results.
    """

    # NOTE: Should do copy here
    # Thorvald (19.09.22): Why?
    arr_roll = arr[:, [*range(arr.shape[1]), *range(arr.shape[1] - 1)]]  # .copy()
    strd_0, strd_1 = arr_roll.strides
    n = arr.shape[1]
    # Set as_strided to writable=False to avoid accidentally writing to the memory
    # thus corrupting the data.
    # Setting this is also recommended in the documentation.
    result = as_strided(
        arr_roll, (*arr.shape, n), (strd_0, strd_1, strd_1), writeable=False
    )

    return result[np.arange(arr.shape[0]), (n - m) % n].astype(arr.dtype)


def _take_per_row_strided(A, start_idx, n_elem):
    """Select n_elem per row with a shift start_idx.

    Fast implementation of selection from row wise shifted sample.
    Rows are not wrapped around, i.e. if start_idx + n_elem is larger than the
    number of columns, out of range is thrown.

    In other words
    >>> def simple_row_strided(A, shift_array, number_elements):
    >>>     strided_A = np.empty((A.shape[0], number_elements))
    >>>     for i in range(A.shape[0]):
    >>>         strided_A[i] = A[i, shift_array[i]:shift_array[i]+number_elements]
    """

    m, n = np.shape(A)
    A.shape = -1
    s0 = A.strides[0]
    l_indx = start_idx + n * np.arange(len(start_idx))
    out = as_strided(A, (len(A) - n_elem + 1, n_elem), (s0, s0), writeable=False)[
        l_indx
    ]
    A.shape = m, n

    return out


class SCMF(BaseMF):
    """Shifted matrix factorization with L2 and convolutional regularization (optional).

    Factor updates are based on gradient descent approximations, permitting an arbitrary
    weight matrix in the discrepancy term. The shift mechanism will maximize the
    correlation between vector samples in the original and estimated data matrices for
    more accurate factor estimates.

    Args:
        X: Sparse data matrix used to estimate factor matrices
        V: Initial estimate for basic vectors
        config: Configuration model.

    Discussion on internal matrices:
        There are four X matrices (correspondingly for W):

        - X : The original input matrix
        - X_bc : The original input matrix with padded zeros on the time axis.
        - X_shifted : Similar to X_bc, but each row is shifted according to self.s.
        - X_shifts : A stack of size(s_budged) arrays similar to X_bc, but each shifted
            horizontally (time axis). Stack layer i is shifted s_budged[i].

        X_shifted is the only matrix altered after initialization.

    """

    def __init__(
        self,
        X,
        config: ModelConfig,
    ):

        self.config = config
        self.W = self.config.weight_matrix_getter(X)

        V = config.initial_basic_profiles_getter(X.shape[1], config.rank)
        self.N, self.T = np.shape(X)
        self.nz_rows, self.nz_cols = np.nonzero(X)

        self.n_iter_ = 0

        # The shift amount per row
        self.s = np.zeros(self.N, dtype=int)
        # The number of possible shifts. Used for padding of arrays.
        self.Ns = len(self.config.shift_budget)

        # Add time points to cover extended left and right boundaries when shifting.
        self.KD = tf.cast(
            self.config.difference_matrix_getter(self.T + 2 * self.Ns), dtype=tf.float32
        )

        self.I1 = self.config.lambda1 * np.identity(self.config.rank)
        self.I2 = self.config.lambda2 * np.identity(self.config.rank)

        # Expand matrices with zeros over the extended left and right boundaries.
        self.X_bc = np.hstack(
            [np.zeros((self.N, self.Ns)), X, np.zeros((self.N, self.Ns))]
        )
        self.W_bc = np.hstack(
            [np.zeros((self.N, self.Ns)), self.W, np.zeros((self.N, self.Ns))]
        )
        self.V_bc = np.vstack(
            [
                np.zeros((self.Ns, self.config.rank)),
                V,
                np.zeros((self.Ns, self.config.rank)),
            ]
        )
        # We know V_bc to be two-dimensional, so cast to please mypy.
        J_shape = cast(tuple[int, int], self.V_bc.shape)
        # TODO: do we have to cast this to tf float32?
        self.J = self.config.minimal_value_matrix_getter(J_shape)

        # Implementation shifts W and Y (not UV.T)
        self.X_shifted = self.X_bc.copy()
        self.W_shifted = self.W_bc.copy()
        self._fill_boundary_regions_V_bc()

        # Placeholders (s x N x T) for all possible candidate shits
        self.X_shifts = np.empty((self.Ns, *self.X_bc.shape))
        self.W_shifts = np.empty((self.Ns, *self.W_bc.shape))

        # Shift Y in opposite direction of V shift.
        for j, s_n in enumerate(self.config.shift_budget):

            self.X_shifts[j] = np.roll(self.X_bc, -1 * s_n, axis=1)
            self.W_shifts[j] = np.roll(self.W_bc, -1 * s_n, axis=1)

        self.U = self._exactly_solve_U()

    @property
    def X(self):
        # return self.X_bc[:, self.Ns:-self.Ns]
        return _take_per_row_strided(self.X_shifted, self.Ns - self.s, n_elem=self.T)

    @property
    def V(self):
        """To be compatible with the expectation of having a V"""
        return self.V_bc

    @property
    def M(self):

        # Compute the reconstructed matrix with sample-specific shifts
        M = _take_per_row_strided(
            self.U @ self.V_bc.T, start_idx=self.Ns - self.s, n_elem=self.T
        )

        return np.array(M, dtype=np.float32)

    def _shift_X_W(self):

        self.X_shifted = _custom_roll(self.X_bc, -1 * self.s)
        self.W_shifted = _custom_roll(self.W_bc, -1 * self.s)

    def _fill_boundary_regions_V_bc(self):
        """Extrapolate the edge values in V_bc over the extended boundaries"""

        V_filled = np.zeros_like(self.V_bc)

        idx = np.arange(self.T + 2 * self.Ns)
        for i, v in enumerate(self.V_bc.T):

            v_left = v[idx <= int(self.T / 2)]
            v_right = v[idx > int(self.T / 2)]

            v_left[v_left == 0] = v_left[np.argmax(v_left != 0)]
            v_right[v_right == 0] = v_right[np.argmax(np.cumsum(v_right != 0))]

            V_filled[:, i] = np.concatenate([v_left, v_right])

        self.V_bc = V_filled

    def _update_V(self):
        V = tf.Variable(self.V_bc, dtype=tf.float32)

        # @tf.function
        def _loss_V():

            frob_tensor = tf.multiply(
                self.W_shifted, self.X_shifted - (self.U @ tf.transpose(V))
            )
            frob_loss = tf.reduce_sum(frob_tensor**2)

            l2_loss = self.config.lambda2 * tf.reduce_sum((V - self.J) ** 2)
            conv_loss = self.config.lambda3 * tf.reduce_sum(
                (tf.matmul(self.KD, V) ** 2)
            )

            return frob_loss + l2_loss + conv_loss

        optimiser = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        for _ in tf.range(self.config.iter_V):
            optimiser.minimize(_loss_V, [V])

        self.V_bc = V.numpy()

    def _exactly_solve_U(self):
        """Solve for U at a fixed V.

        The internal U member is not modified by this method.
        V is assumed to be initialized."""
        U = np.empty((self.N, self.config.rank))

        for n in range(self.N):
            U[n] = (
                self.X_shifted[n]
                @ self.V_bc
                @ np.linalg.inv(
                    self.V_bc.T @ (np.diag(self.W_shifted[n]) @ self.V_bc) + self.I1
                )
            )
        return U

    def _approx_U(self):

        U = tf.Variable(self.U, dtype=tf.float32)

        # @tf.function
        def _loss_U():

            frob_tensor = tf.multiply(
                self.W_shifted,
                self.X_shifted - tf.matmul(U, self.V_bc, transpose_b=True),
            )
            frob_loss = tf.reduce_sum((frob_tensor) ** 2)

            return frob_loss + self.config.lambda1 * tf.reduce_sum(U**2)

        optimiser = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        for _ in tf.range(self.config.iter_U):
            optimiser.minimize(_loss_U, [U])

        return U.numpy()

    def _update_s(self):

        # Evaluate the discrepancy term for all possible shift candidates
        M = self.U @ self.V_bc.T
        D = (
            np.linalg.norm(self.W_shifts * (self.X_shifts - M[None, :, :]), axis=-1)
            ** 2
        )

        # Selected shifts maximize the correlation between X and M
        s_new = [self.config.shift_budget[i] for i in np.argmin(D, axis=0)]

        # Update attributes only if changes to the optimal shift
        if not np.array_equal(self.s, s_new):

            self.s = np.array(s_new)
            self._shift_X_W()

    def run_step(self):
        "Perform one step of alternating minimization"

        self.U = self._approx_U()
        self._update_V()
        self._update_s()

        self.n_iter_ += 1

    def loss(self):
        "Compute the loss from the optimization objective"

        loss = np.sum(
            np.linalg.norm(
                self.W_shifted * (self.X_shifted - self.U @ self.V_bc.T), axis=1
            )
            ** 2
        )
        loss += self.config.lambda1 * np.sum(np.linalg.norm(self.U, axis=1) ** 2)
        loss += self.config.lambda2 * np.linalg.norm(self.V_bc) ** 2
        loss += self.config.lambda3 * np.linalg.norm(self.KD @ self.V_bc) ** 2

        return loss
