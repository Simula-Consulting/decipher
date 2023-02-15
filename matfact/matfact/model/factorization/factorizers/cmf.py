from warnings import warn

import numpy as np

from matfact.model.config import ModelConfig

from .mfbase import BaseMF


class CMF(BaseMF):
    """Matrix factorization with L2 and convolutional regularization.
    Factor updates are based on analytical estimates and will therefore
    not permit and arbitrary weight matrix in the discrepancy term.

    Args:
            X: Sparse data matrix used to estimate factor matrices
            V: Initial estimate for basic vectors
            config: Configuration model.
              shift and weights are ignored in the CMF factorizer.
    """

    def __init__(
        self,
        X,
        config: ModelConfig,
    ):
        if not config.weight_matrix_getter.is_identity:
            warn(
                "CMF given a non-identity weight. This will be ignored."
                "Consider using WCMF or SCMF."
            )
        if config.shift_budget:
            warn(
                "CMF given a non-empty shift budget. This will be ignored."
                "Consider using SCMF."
            )

        self.X = X
        self.V = config.initial_basic_profiles_getter(X.shape[1], config.rank)
        self.config = config

        self.N, self.T = np.shape(self.X)
        self.nz_rows, self.nz_cols = np.nonzero(self.X)

        self.n_iter_ = 0
        KD = self.config.difference_matrix_getter(self.T)
        self.J = self.config.minimal_value_matrix_getter((self.T, self.config.rank))
        self._init_matrices(KD)
        # Initialize U
        self.U = np.zeros((self.N, config.rank))
        self._update_U()

    @property
    def M(self):
        return np.array(self.U @ self.V.T, dtype=np.float32)

    def _init_matrices(self, KD):
        self.S = self.X.copy()
        self.mask = (self.X != 0).astype(np.float32)

        self.lambda1_l1= self.config.U_l1_rate * self.config.lambda1  # U l1 regularization parameter
        self.lambda1_l2= self.config.lambda1 - self.lambda1_l1  # U l2 regularization parameter
        self.I_lambda1_l2 = self.lambda1_l2 * np.identity(self.config.rank)  # U l2 regularization
        self.I_lambda2_l2 = self.config.lambda2 * np.identity(self.config.rank)  # V l2 regarization

        self.KD = KD  # K := Toeplitz Matrix, D := Forward Difference Matrix
        self.DTKTKD = KD.T @ KD
        self.L2, self.Q2 = np.linalg.eigh(self.config.lambda3 * self.DTKTKD)  # V temporal smoothness regularization

    def _update_V(self):

        L1, Q1 = np.linalg.eigh(self.U.T @ self.U + self.I_lambda2_l2)

        hatV = (
            (self.Q2.T @ (self.S.T @ self.U + self.config.lambda2 * self.J))
            @ Q1
            / np.add.outer(self.L2, L1)
        )
        self.V = self.Q2 @ (hatV @ Q1.T)

    def _update_U(self):
        self.U = np.transpose(
            np.linalg.solve(
                self.V.T @ self.V + self.I_lambda1_l2,
                self.V.T @ self.S.T - self.lambda1_l1 * np.sign(self.U.T),
            )
        )

    def _update_S(self):
        self.S = self.U @ self.V.T
        self.S[self.nz_rows, self.nz_cols] = self.X[self.nz_rows, self.nz_cols]

    def U_l1_reg_loss(self):
        return self.lambda1_l1 * np.linalg.norm(self.U)

    def U_l2_reg_loss(self):
        return self.lambda1_l2 * np.square(np.linalg.norm(self.U))

    def V_l2_reg_loss(self):
        return self.config.lambda2 * np.square(np.linalg.norm(self.V - self.J))

    def V_R_reg_loss(self):
        return self.config.lambda3 * np.square(np.linalg.norm(self.KD @ self.V))

    def loss(self):
        "Compute the loss from the optimization objective"

        loss = np.square(np.linalg.norm(self.mask * (self.X - self.U @ self.V.T)))
        loss += self.U_l1_reg_loss() + self.U_l2_reg_loss()
        loss += self.V_l2_reg_loss() + self.V_R_reg_loss()

        return loss

    def run_step(self):
        "Perform one step of alternating minimization"

        self._update_U()
        self._update_V()
        self._update_S()

        self.n_iter_ += 1
