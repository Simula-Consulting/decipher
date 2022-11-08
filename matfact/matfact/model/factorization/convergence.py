import numpy as np
from tqdm import trange

from matfact import settings


def convergence_monitor(M, error_tol=1e-4):
    """Track convergence of the matrix completion process by measuring the
    difference between consecutive estimates.
    """
    return MonitorFactorUpdate(M=M, tol=error_tol)


class ConvergenceMonitor:
    def __init__(
        self,
        number_of_epochs=settings.default_number_of_epochs,
        epochs_per_val=settings.default_epochs_per_val,
        patience=settings.default_patience,
        show_progress=True,
        tolerance=1e-4,
    ):
        self.number_of_epochs = number_of_epochs
        self.tolerance = tolerance
        self.epochs_per_val = epochs_per_val
        self.patience = patience
        self._old_M = None
        self._model = None
        self._range = trange if show_progress else range

    def _update(self):
        new_M = self._model.M
        difference = float(
            np.linalg.norm(new_M - self._old_M) ** 2 / np.linalg.norm(self._old_M) ** 2
        )
        self._old_M = new_M
        return difference

    def __call__(self, model):
        self._old_M = model.X  # Dirty hack.
        self._model = model
        for i in self._range(self.number_of_epochs):
            yield i
            should_update = i > self.patience and i % self.epochs_per_val == 0
            if should_update and self._update() < self.tolerance:
                break


class MonitorFactorUpdate:
    def __init__(self, M, tol=1e-6):

        self.M = M
        self.tol = tol

        self.n_iter_ = 0
        self.update_ = []
        self.convergence_rate_ = []

    def _should_stop(self, M_new):

        update = float(
            np.linalg.norm(M_new - self.M) ** 2 / np.linalg.norm(self.M) ** 2
        )

        if np.isnan(update):
            raise ValueError("Update value is NaN")

        self.update_.append(update)

        return update < self.tol

    def track_convergence_rate(self, M_new):

        self.Mpp = self.Mp
        self.Mp = self.M
        self.M = M_new

        a = np.linalg.norm(self.M - self.Mp)
        b = np.linalg.norm(self.Mp - self.Mpp) + 1e-12
        self.convergence_rate_.append(a / b)

    def converged(self, M_new):

        should_stop = self._should_stop(M_new)

        if self.n_iter_ > 0:
            self.track_convergence_rate(M_new=M_new)

        else:
            self.Mp = self.M
            self.M = M_new

        self.n_iter_ += 1

        return should_stop
