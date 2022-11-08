import numpy as np
import tensorflow as tf

from matfact.model.config import ModelConfig
from matfact.model.factorization.weights import data_weights

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
        W=None,
        D=None,
        J=None,
        K=None,
    ):
        self.X = X
        self.V = V
        self.W = data_weights(X) if W is None else W

        self.config = config

        self.r = V.shape[1]
        self.N, self.T = np.shape(self.X)
        self.nz_rows, self.nz_cols = np.nonzero(self.X)

        self.n_iter_ = 0
        KD = self.config.difference_matrix_getter(self.T)
        self.J = self.config.minimal_value_matrix_getter((self.T, self.r))
        self._init_matrices(KD)

    @property
    def M(self):
        return np.array(self.U @ self.V.T, dtype=np.float32)

    def _init_matrices(self, KD):

        self.KD = tf.cast(KD, dtype=tf.float32)
        self.DTKTKD = KD.T @ KD

        self.I_l1 = self.config.lambda1 * np.eye(self.r)

    def _update_V(self):
        # @tf.function
        def _loss_V():
            frob_tensor = tf.multiply(W, X - (U @ tf.transpose(V)))
            frob_loss = tf.square(tf.norm(frob_tensor))

            l2_loss = self.config.lambda2 * tf.square(tf.norm(V - J))

            conv_loss = self.config.lambda3 * tf.square(tf.norm(tf.matmul(self.KD, V)))

            return frob_loss + l2_loss + conv_loss

        V = tf.Variable(self.V, dtype=tf.float32)
        J = tf.ones_like(self.V, dtype=tf.float32)

        W = tf.cast(self.W, dtype=tf.float32)
        X = tf.cast(self.X, dtype=tf.float32)
        U = tf.cast(self.U, dtype=tf.float32)

        optimiser = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        for _ in tf.range(self.config.iter_V):
            optimiser.minimize(_loss_V, [V])

        self.V = V.numpy()

    def _approx_U(self):
        # @tf.function
        def _loss_U():
            frob_tensor = tf.multiply(W, X - tf.matmul(U, V, transpose_b=True))
            frob_loss = tf.square(tf.norm(frob_tensor))

            return frob_loss + self.config.lambda1 * tf.square(tf.norm(U))

        U = tf.Variable(self.U, dtype=tf.float32)

        W = tf.cast(self.W, dtype=tf.float32)
        X = tf.cast(self.X, dtype=tf.float32)
        V = tf.cast(self.V, dtype=tf.float32)

        optimiser = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)

        for _ in tf.range(self.config.iter_U):
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
        loss += self.config.lambda1 * np.square(np.linalg.norm(self.U))
        loss += self.config.lambda2 * np.square(np.linalg.norm(self.V - 1))
        loss += self.config.lambda3 * np.square(np.linalg.norm(self.KD @ self.V))

        return loss

    def run_step(self):
        "Perform one step of alternating minimization"

        self._update_U()
        self._update_V()

        self.n_iter_ += 1
