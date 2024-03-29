from abc import ABC, abstractmethod

import numpy.typing as npt

from matfact.model.factorization.convergence import ConvergenceMonitor, EpochGenerator
from matfact.model.factorization.utils import theta_mle
from matfact.model.predict.risk_prediction import predict_proba


class BaseMF(ABC):
    "Base class for matrix factorization algorithms."

    X: npt.NDArray
    U: npt.NDArray
    V: npt.NDArray

    @property
    @abstractmethod
    def M(self):
        ...

    @abstractmethod
    def run_step(self):
        return

    @abstractmethod
    def loss(self):
        return

    def predict_probability(self, observed_data, t_pred):
        """Predict the probability of the possible states at t_pred"""
        return predict_proba(
            observed_data,
            self.M,
            t_pred,
            theta_mle(self.X, self.M),
            number_of_states=self.config.number_of_states,
        )

    def matrix_completion(
        self,
        extra_metrics=None,
        epoch_generator: EpochGenerator | None = None,
    ):
        """Run matrix completion on input matrix X using a factorization model.

        Arguments:
            extra_metrics: Dict of name, callable pairs for extra metric logging.
                Callable must have the signature `(model: Type[BaseMF]) -> Float`.
            epoch_generator:  A generator of epoch numbers. Defaults to
                [`ConvergenceMonitor`][matfact.model.factorization.convergence.ConvergenceMonitor]
                which implements eager termination if detecting convergence.
        """
        if epoch_generator is None:
            epoch_generator = ConvergenceMonitor()

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

        if extra_metrics is None:
            extra_metrics = {}

        for metric in extra_metrics:
            output[metric] = []

        for epoch in epoch_generator(self):

            self.run_step()

            output["epochs"].append(int(epoch))
            output["loss"].append(float(self.loss()))
            for metric, callable in extra_metrics.items():
                output[metric].append(callable(self))

        output["U"] = self.U
        output["V"] = self.V
        output["M"] = self.M
        output["s"] = getattr(self, "s", None)

        output["theta_mle"] = theta_mle(self.X, self.M)

        return output
