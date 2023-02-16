import numpy as np
import numpy.typing as npt
from sklearn.decomposition import NMF, DictionaryLearning, TruncatedSVD

from matfact.experiments.config import FactorizerType, SklearnFactorizerConfig
from matfact.model.factorization.utils import theta_mle
from matfact.model.predict.risk_prediction import predict_proba


def sklearn_model_factory(
    X: np.ndarray,
    rank: int = 5,
    **kwargs,
):
    """
    Initialize and return appropriate scikit-learn model based on arguments.

    kwargs are passed the SklearnFactorizerConfig.
    """
    config = SklearnFactorizerConfig(rank=rank, **kwargs)

    return SklearnFactorizer(X, config)


class SklearnFactorizer:
    """
    Class that allows use of sklearn matrix
    factorization models as MatFact factorizers.
    """

    def __init__(
        self,
        observation_matrix: npt.NDArray[np.int_],
        config: SklearnFactorizerConfig,
    ) -> None:
        # Initiate factorizer
        self.X = observation_matrix
        self.N, self.T = np.shape(self.X)
        self.nz_rows, self.nz_cols = np.nonzero(self.X)

        self._initiate_factorizer(config)
        self._fit()

    @property
    def M(self):
        return np.array(self.U @ self.V.T, dtype=np.float32)

    def _initiate_factorizer(self, config: SklearnFactorizerConfig) -> None:
        # instantiate the chosen model-type with corresponding hyperparameters
        if config.model_type is FactorizerType.STSVD:
            self._factorizer = TruncatedSVD(
                n_components=config.rank,
                algorithm="randomized",
                n_iter=100,
                random_state=0,
            )
        elif config.model_type is FactorizerType.SDL:
            self._factorizer = DictionaryLearning(
                n_components=config.rank,
                alpha=0,
                positive_code=False,
                positive_dict=False,
                fit_algorithm="cd",
                transform_algorithm="lasso_cd",
                random_state=0,
            )
        elif config.model_type is FactorizerType.SNMF:
            self._factorizer = NMF(
                n_components=config.rank,
                init="random",
                alpha_W=config.lambda1,  # U reg
                alpha_H=config.lambda2,  # V reg
                l1_ratio=0,  # config.U_l1_rate,
                max_iter=500,
                random_state=0,
            )
        else:
            raise ValueError(
                f"config.model_type has to be one of {FactorizerType.STSVD}, {FactorizerType.SDL} or {FactorizerType.SNMF}!"
            )

    def _fit(self) -> None:
        """Fit the model."""
        self._factorizer.fit(self.X)
        self.U = self._factorizer.transform(self.X)
        self.V = self._factorizer.components_.T  # <- V

    def _fit_transform(self) -> npt.NDArray:
        """Fit the model and tranform."""
        return self._factorizer.fit_transform(self.X)

    def _transform(self) -> npt.NDArray:
        """Transform."""
        return self._factorizer.transform(self.X)

    def matrix_completion(self):
        # Store model matrices
        # Results collected from the process
        output: dict = {
            "loss": [],
            "epochs": [],
            "U": None,
            "V": None,
            "M": None,
            "s": None,
            "theta_mle": None,
        }
        output["U"] = self.U
        output["V"] = self.V
        output["M"] = self.M
        output["theta_mle"] = theta_mle(self.X, self.M)

        return output

    def predict_probability(self, observed_data, t_pred):
        """Predict the probability of the possible states at t_pred"""
        # U = self._factorizer.transform(observed_data)
        # M = np.array(U @ self.V.T, dtype=np.float32)
        return predict_proba(
            observed_data,
            self.M,
            t_pred,
            theta_mle(self.X, self.M),
            number_of_states=self.config.number_of_states,
        )
