import numpy as np
import tensorflow as tf
from numpy.lib.stride_tricks import as_strided

from .mfbase import BaseMF


def _custom_roll(arr, m):
    # Auxillary function for faster row-wise shifting

    # NOTE: Should do copy here
    arr_roll = arr[:, [*range(arr.shape[1]), *range(arr.shape[1] - 1)]].copy()
    strd_0, strd_1 = arr_roll.strides
    n = arr.shape[1]
    result = as_strided(arr_roll, (*arr.shape, n), (strd_0, strd_1, strd_1))

    return result[np.arange(arr.shape[0]), (n - m) % n].astype(arr.dtype)


def _take_per_row_strided(A, start_idx, n_elem):
    # Auxillary function for faster selecting from shifted samples

    m, n = np.shape(A)
    A.shape = -1
    s0 = A.strides[0]
    l_indx = start_idx + n * np.arange(len(start_idx))
    out = as_strided(A, (len(A) - n_elem + 1, n_elem), (s0, s0))[l_indx]
    A.shape = m, n

    return out


class SCMF(BaseMF):
    """Shifted matrix factorization with L2 and convolutional regularization (optional).
        Factor updates are based on gradient descent approximations, permitting
        an arbitrary weight matrix in the discrepancy term. The shift mechanism will maximize the
    correlation between vector samples in the original and estimated data matrices for more accurate
    factor estimates.

    Args:
                X: Sparse data matrix used to estimate factor matrices
                V: Initial estimate for basic vectors
        s_budget: Range of possible shift steps (forward or backward from the original position)
                W (optional): Weight matrix for the discrepancy term
                D (optional): Forward difference matrix
                J (optional): A martix used to impose a minimum value in the basic vecors V
                K (optional): Convolutional matrix
                lambda: Regularization coefficients
                iter_U, iter_V: The number of steps with gradient descent (GD) per factor update
                learning_rate: Stepsize used in the GD
    """

    def __init__(
        self,
        X,
        V,
        s_budget,
        W=None,
        D=None,
        J=None,
        K=None,
        lambda1=1.0,
        lambda2=1.0,
        lambda3=1.0,
        iter_U=2,
        iter_V=2,
        learning_rate=0.001,
    ):

        self.X = X
        self.V = V
        self.W = W

        self.s_budget = s_budget

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

        self.iter_U = iter_U
        self.iter_V = iter_V
        self.learning_rate = learning_rate

        self.r = V.shape[1]
        self.N, self.T = np.shape(self.X)
        self.nz_rows, self.nz_cols = np.nonzero(self.X)

        self.n_iter_ = 0
        self._init_matrices(D, J, K)

    @property
    def M(self):

        # Compute the reconstructed matrix with sample-specific shifts
        M = _take_per_row_strided(
            self.U @ self.V.T, start_idx=self.Ns - self.s, n_elem=self.T
        )

        return np.array(M, dtype=np.float32)

    def _init_matrices(self, D, J, K):

        self.s = np.zeros(self.N, dtype=int)
        self.Ns = int(self.s_budget.size)

        # Add time points to cover extended left and right boundaries when shifting.
        self.K = np.eye(self.T + 2 * self.Ns) if K is None else K
        self.D = np.eye(self.T + 2 * self.Ns) if D is None else D
        self.KD = tf.cast(self.K @ self.D, dtype=tf.float32)

        self.I1 = self.lambda1 * np.identity(self.r)
        self.I2 = self.lambda2 * np.identity(self.r)

        # Expand matrices with zeros over the extended left and right boundaries.
        self.X_bc = np.hstack(
            [np.zeros((self.N, self.Ns)), self.X, np.zeros((self.N, self.Ns))]
        )
        self.W_bc = np.hstack(
            [np.zeros((self.N, self.Ns)), self.W, np.zeros((self.N, self.Ns))]
        )

        self.V = np.vstack(
            [np.zeros((self.Ns, self.r)), self.V, np.zeros((self.Ns, self.r))]
        )

        # Placeholders (s x N x T) for all possible candidate shits
        self.X_shifts = np.array([np.zeros_like(self.X_bc)] * self.Ns)
        self.W_shifts = np.array([np.zeros_like(self.W_bc)] * self.Ns)

        # Implementation shifts W and Y (not UV.T)
        self._shift_X_W()
        self._fill_boundary_regions_V()

        # Shift Y in opposite direction of V shift.
        for j, s_n in enumerate(self.s_budget):

            self.X_shifts[j] = np.roll(self.X_bc, -1 * s_n, axis=1)
            self.W_shifts[j] = np.roll(self.W_bc, -1 * s_n, axis=1)

    def _shift_X_W(self):

        self.X = _custom_roll(self.X_bc.copy(), -1 * self.s)
        self.W = _custom_roll(self.W_bc.copy(), -1 * self.s)

    def _fill_boundary_regions_V(self):
        # Extrapolate the edge values in V over the extended boundaries

        V_filled = np.zeros_like(self.V)

        idx = np.arange(self.T + 2 * self.Ns)
        for i, v in enumerate(self.V.T):

            v_left = v[idx <= int(self.T / 2)]
            v_right = v[idx > int(self.T / 2)]

            v_left[v_left == 0] = v_left[np.argmax(v_left != 0)]
            v_right[v_right == 0] = v_right[np.argmax(np.cumsum(v_right != 0))]

            V_filled[:, i] = np.concatenate([v_left, v_right])

        self.V = V_filled

    def _update_V(self):

        # @tf.function
        def _loss_V():

            frob_tensor = tf.multiply(W, X - (U @ tf.transpose(V)))
            frob_loss = tf.reduce_sum(tf.square(tf.norm(frob_tensor, axis=-1)))

            l2_loss = self.lambda2 * tf.square(tf.norm(V - J))
            conv_loss = self.lambda3 * tf.square(tf.norm(tf.matmul(self.KD, V)))

            return frob_loss + l2_loss + conv_loss

        V = tf.Variable(self.V, dtype=tf.float32)
        J = tf.ones_like(self.V, dtype=tf.float32)

        W = tf.cast(self.W, dtype=tf.float32)
        X = tf.cast(self.X, dtype=tf.float32)
        U = tf.cast(self.U, dtype=tf.float32)

        optimiser = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        for _ in tf.range(self.iter_V):
            optimiser.minimize(_loss_V, [V])

        self.V = V.numpy()

    def _approx_U(self):

        # @tf.function
        def _loss_U():

            frob_tensor = tf.multiply(W, X - tf.matmul(U, V, transpose_b=True))
            frob_loss = tf.reduce_sum(tf.square(tf.norm(frob_tensor, axis=-1)))

            return frob_loss + self.lambda1 * tf.reduce_sum(
                tf.square(tf.norm(U, axis=-1))
            )

        U = tf.Variable(self.U, dtype=tf.float32)

        W = tf.cast(self.W, dtype=tf.float32)
        X = tf.cast(self.X, dtype=tf.float32)
        V = tf.cast(self.V, dtype=tf.float32)

        optimiser = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        for _ in tf.range(self.iter_U):
            optimiser.minimize(_loss_U, [U])

        return U.numpy()

    def _update_U(self):

        # Faster to approximate U in consecutive iterations
        if self.n_iter_ > 0:
            self.U = self._approx_U()

        else:

            # Estimate U in the first iteration of alternating minimization
            self.U = np.zeros((self.N, self.r))

            for n in range(self.N):
                self.U[n] = (
                    self.X[n]
                    @ self.V
                    @ np.linalg.inv(self.V.T @ (np.diag(self.W[n]) @ self.V) + self.I1)
                )

    def _update_s(self):

        # Evaluate the discrepancy term for all possible shift candidates
        M = self.U @ self.V.T
        D = (
            np.linalg.norm(self.W_shifts * (self.X_shifts - M[None, :, :]), axis=-1)
            ** 2
        )

        # Selected shifts maximize the correlation between X and M
        s_new = self.s_budget[np.argmin(D, axis=0)]

        # Update attributes only if changes to the optimal shift
        if not np.array_equal(self.s, s_new):

            self.s = s_new
            self._shift_X_W()

    def run_step(self):
        "Perform one step of alternating minimization"

        self._update_U()
        self._update_V()
        self._update_s()

        self.n_iter_ += 1

    def loss(self):
        "Compute the loss from the optimization objective"

        loss = np.sum(
            np.linalg.norm(self.W * (self.X - self.U @ self.V.T), axis=1) ** 2
        )
        loss += self.lambda1 * np.sum(np.linalg.norm(self.U, axis=1) ** 2)
        loss += self.lambda2 * np.linalg.norm(self.V) ** 2
        loss += self.lambda3 * np.linalg.norm(self.KD @ self.V) ** 2

        return loss
