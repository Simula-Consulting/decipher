import numpy as np
import tensorflow as tf

from matfact.config import ModelConfig, ParameterConfig

from ...simulation import data_weights
from .mfbase import BaseMF


class WCMF(BaseMF):
    """Matrix factorization with L2 and convolutional regularization.
    Factor updates are based on gradient descent approximations, permitting
    an arbitrary weight matrix in the discrepancy term.

    Args:
        X: Sparse data matrix used to estimate factor matrices
        V: Initial estimate for basic vectors
        W (optional): Weight matrix for the discrepancy term. Default is
            set to the output of `experiments.simulation.data_weights(X)`.
        D (optional): Forward difference matrix
        J (optional): A matrix used to impose a minimum value in the basic vectors V
        K (optional): Convolutional matrix
        lambda: Regularization coefficients
        iter_U, iter_V: The number of steps with gradient descent (GD) per factor update
        learning_rate: Step size used in the GD
    """

    def __init__(
        self,
        X,
        V,
        config: ModelConfig,
        parameters: ParameterConfig,
    ):
        _weights_getter = config.weights_getter or data_weights

        self.X = X
        self.V = V
        self.W = _weights_getter(X)

        self.number_of_states = parameters.number_of_states

        self.lambda1 = parameters.lambda1
        self.lambda2 = parameters.lambda2
        self.lambda3 = parameters.lambda3

        self.iter_U = parameters.iter_U
        self.iter_V = parameters.iter_V
        self.learning_rate = parameters.learning_rate

        self.r = V.shape[1]
        self.N, self.T = np.shape(self.X)
        self.nz_rows, self.nz_cols = np.nonzero(self.X)

        self.n_iter_ = 0
        self._init_matrices(
            config.differential_matrix_getter,
            config.minimum_values,
        )

    @property
    def M(self):
        return np.array(self.U @ self.V.T, dtype=np.float32)

    def _init_matrices(self, KD_getter, J):
        self.J = np.ones((self.T, self.r)) if J is None else J

        self.KD = tf.cast(KD_getter(self.T), dtype=tf.float32)
        self.DTKTKD = (self.KD.numpy()).T @ (self.KD.numpy())

        self.I_l1 = self.lambda1 * np.eye(self.r)

    def _update_V(self):
        # @tf.function
        def _loss_V():
            frob_tensor = tf.multiply(W, X - (U @ tf.transpose(V)))
            frob_loss = tf.square(tf.norm(frob_tensor))

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
            frob_loss = tf.square(tf.norm(frob_tensor))

            return frob_loss + self.lambda1 * tf.square(tf.norm(U))

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
                    self.V.T
                    @ (self.W[n] * self.X[n])
                    @ np.linalg.inv(
                        self.V.T @ (self.W[n][:, None] * self.V) + self.I_l1
                    )
                )

    def loss(self):
        "Compute the loss from the optimization objective"

        loss = np.square(np.linalg.norm(self.W * (self.X - self.U @ self.V.T)))
        loss += self.lambda1 * np.square(np.linalg.norm(self.U))
        loss += self.lambda2 * np.square(np.linalg.norm(self.V - 1))
        loss += self.lambda3 * np.square(np.linalg.norm(self.KD @ self.V))

        return loss

    def run_step(self):
        "Perform one step of alternating minimization"

        self._update_U()
        self._update_V()

        self.n_iter_ += 1
