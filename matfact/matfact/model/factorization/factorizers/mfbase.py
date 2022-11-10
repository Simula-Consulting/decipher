from abc import ABC, abstractmethod

from tqdm import tqdm

from matfact.model.factorization.convergence import convergence_monitor
from matfact.model.factorization.utils import theta_mle
from matfact.model.predict.risk_prediction import predict_proba


class BaseMF(ABC):
    "Base class for matrix factorization algorithms."

    @property
    @abstractmethod
    def M(self):
        return

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
        fname="",
        epochs_per_val=5,
        num_epochs=2000,
        patience=200,
        progress=True,
    ):
        """Run matrix completion on input matrix X using a factorization model.

        extra_metrics: Dict of name, callable pairs for extra metric logging.
            Callable must have the signature (model: Type[BaseMF]) -> Float.
        """

        # Results collected from the process
        output = {
            "convergence_rate": [],
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

        for epoch in tqdm(range(num_epochs), disable=not progress, desc="Epoch: "):

            self.run_step()

            output["epochs"].append(int(epoch))
            output["loss"].append(float(self.loss()))
            for metric, callable in extra_metrics.items():
                output[metric].append(callable(self))

            if epoch == patience:
                monitor = convergence_monitor(self.M)

            if epoch % epochs_per_val == 0 and epoch > patience:

                if monitor.converged(self.M):
                    break

        output["U"] = self.U
        output["V"] = self.V
        output["M"] = self.M

        if hasattr(self, "s"):
            output["s"] = self.s

        output["theta_mle"] = theta_mle(self.X, self.M)

        return output
