import numpy as np

from matfact import settings
from matfact.config import ModelConfig

from .mfbase import BaseMF


class CMF(BaseMF):
    """Matrix factorization with L2 and convolutional regularization.
    Factor updates are based on analytical estimates and will therefore
    not permit and arbitrary weight matrix in the discrepancy term.

    Args:
            X: Sparse data matrix used to estimate factor matrices
            V: Initial estimate for basic vectors
            D (optional): Forward difference matrix
            J (optional): A matrix used to impose a minimum value in the basic vectors V
            K (optional): Convolutional matrix
            lambda: Regularization coefficients
    """

    def __init__(
        self,
        X,
        V,
        config: ModelConfig,
        lambda1=1.0,
        lambda2=1.0,
        lambda3=1.0,
        number_of_states: int = settings.default_number_of_states,
    ):

        self.X = X
        self.V = V

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

        self.r = V.shape[1]
        self.N, self.T = np.shape(self.X)
        self.nz_rows, self.nz_cols = np.nonzero(self.X)

        self.n_iter_ = 0
        self._init_matrices(
            config.differential_matrix,
            config.minimum_values,
        )
        self.number_of_states = number_of_states

    @property
    def M(self):
        return np.array(self.U @ self.V.T, dtype=np.float32)

    def _init_matrices(self, KD, J):

        self.S = self.X.copy()
        self.mask = (self.X != 0).astype(np.float32)

        self.J = np.ones((self.T, self.r)) if J is None else J

        self.KD = np.identity(self.T) if KD is None else KD

        self.I_l1 = self.lambda1 * np.identity(self.r)
        self.I_l2 = self.lambda2 * np.identity(self.r)

        self.DTKTKD = (self.KD).T @ (self.KD)
        self.L2, self.Q2 = np.linalg.eigh(self.lambda3 * self.DTKTKD)

    def _update_V(self):

        L1, Q1 = np.linalg.eigh(self.U.T @ self.U + self.I_l2)

        hatV = (
            (self.Q2.T @ (self.S.T @ self.U + self.lambda2 * self.J))
            @ Q1
            / np.add.outer(self.L2, L1)
        )
        self.V = self.Q2 @ (hatV @ Q1.T)

    def _update_U(self):

        self.U = np.transpose(
            np.linalg.solve(self.V.T @ self.V + self.I_l1, self.V.T @ self.S.T)
        )

    def _update_S(self):

        self.S = self.U @ self.V.T
        self.S[self.nz_rows, self.nz_cols] = self.X[self.nz_rows, self.nz_cols]

    def loss(self):
        "Compute the loss from the optimization objective"

        loss = np.square(np.linalg.norm(self.mask * (self.X - self.U @ self.V.T)))
        loss += self.lambda1 * np.square(np.linalg.norm(self.U))
        loss += self.lambda2 * np.square(np.linalg.norm(self.V - self.J))
        loss += self.lambda3 * np.square(np.linalg.norm(self.KD @ self.V))

        return loss

    def run_step(self):
        "Perform one step of alternating minimization"

        self._update_U()
        self._update_V()
        self._update_S()

        self.n_iter_ += 1
