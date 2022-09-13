from abc import ABC, abstractmethod

from tqdm import tqdm

from ..convergence import convergence_monitor
from ..utils import theta_mle


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

    def matrix_completion(
        self,
        extra_metrics=None,
        fname="",
        epochs_per_val=5,
        num_epochs=2000,
        patience=200,
    ):
        """Run matrix completion on input matrix X using a factorization model.

        extra_metrics: iterable of name, exectuable pairs for extra metric logging.
            iterable must have the signature (model: Type[BaseMF]) -> Float.
        """

        # Results collected from the process
        output = {
            "convergence_rate": [],
            "loss_values": [],
            "epochs": [],
            "U": None,
            "V": None,
            "M": None,
            "s": None,
            "theta_mle": None,
        }

        for metric, _ in extra_metrics:
            output[metric] = []

        for epoch in tqdm(range(num_epochs)):

            self.run_step()

            output["epochs"].append(int(epoch))
            output["loss_values"].append(float(self.loss()))
            for metric, callable in extra_metrics:
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
