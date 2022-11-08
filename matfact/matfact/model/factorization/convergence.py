import numpy as np
from tqdm import trange

from matfact.settings import (
    DEFAULT_EPOCHS_PER_VAL,
    DEFAULT_NUMBER_OF_EPOCHS,
    DEFAULT_PATIENCE,
)


class ConvergenceMonitor:
    def __init__(
        self,
        number_of_epochs=DEFAULT_NUMBER_OF_EPOCHS,
        epochs_per_val=DEFAULT_EPOCHS_PER_VAL,
        patience=DEFAULT_PATIENCE,
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
        # Dirty hack. Model's M is not defined before after first iteration.
        # Therefore, we set the value to X, which we know has the same dimensions.
        self._old_M = model.X
        self._model = model
        for i in self._range(self.number_of_epochs):
            yield i
            should_update = i > self.patience and i % self.epochs_per_val == 0
            if should_update and self._update() < self.tolerance:
                break
