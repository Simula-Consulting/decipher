from collections.abc import Iterator
from typing import Protocol

import numpy as np
from tqdm import trange

from matfact.model import BaseMF
from matfact.settings import (
    DEFAULT_EPOCHS_PER_VAL,
    DEFAULT_NUMBER_OF_EPOCHS,
    DEFAULT_PATIENCE,
)


class EpochGenerator(Protocol):
    def __call__(self, model: BaseMF) -> Iterator[int]:
        ...


class ConvergenceMonitor:
    """Epoch generator with eager termination when converged.

    The model is said to have converged when the model's latent matrix (M) has
    a relative norm difference smaller than tolerance.

    Sample usage:
    >>> monitor = ConvergenceMonitor(tolerance=1e-5)
    >>> for epoch in monitor(model):
    >>>     # If model converges, the generator will deplete before the default number
    >>>     # of epochs has been reached.
    >>>     ...
    """

    def __init__(
        self,
        number_of_epochs: int = DEFAULT_NUMBER_OF_EPOCHS,
        epochs_per_val: int = DEFAULT_EPOCHS_PER_VAL,
        patience: int = DEFAULT_PATIENCE,
        show_progress: bool = True,
        tolerance: float = 1e-4,
    ):
        """Initialize ConvergenceMonitor.

        Args:
          number_of_epochs: the maximum number of epochs to generate.
          epochs_per_val: convergence checking is done every epochs_per_val epoch.
          patience: the minimum number of epcohs.
          show_progress: enable tqdm progress bar.
          tolerance: the tolerance under which the model is said to have converged."""

        self.number_of_epochs = number_of_epochs
        self.tolerance = tolerance
        self.epochs_per_val = epochs_per_val
        self.patience = patience
        self._old_M = None
        self._model = None
        self._range = trange if show_progress else range

    def _update(self):
        new_M = self._model.M
        difference = np.sum((new_M - self._old_M) ** 2) / np.sum(self._old_M**2)
        self._old_M = new_M
        return difference

    def __call__(self, model):
        """A generator that yields epoch numbers."""
        self._old_M = model.M
        self._model = model
        for i in self._range(self.number_of_epochs):
            yield i
            should_update = i > self.patience and i % self.epochs_per_val == 0
            if should_update and self._update() < self.tolerance:
                break
