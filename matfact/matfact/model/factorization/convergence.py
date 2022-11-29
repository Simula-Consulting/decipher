from collections.abc import Iterator
from typing import TYPE_CHECKING, Callable

import numpy as np
from tqdm import trange

from matfact.settings import (
    DEFAULT_EPOCHS_PER_VAL,
    DEFAULT_NUMBER_OF_EPOCHS,
    DEFAULT_PATIENCE,
)

if TYPE_CHECKING:
    from matfact.model import BaseMF


EpochGenerator = Callable[["BaseMF"], Iterator[int]]


class ConvergenceMonitor:
    """Epoch generator with eager termination when converged.

    The model is said to have converged when the model's latent matrix (M) has
    a relative norm difference smaller than tolerance.


    Args:
        number_of_epochs: the maximum number of epochs to generate.
        epochs_per_val: convergence checking is done every epochs_per_val epoch.
        patience: the minimum number of epcohs.
        show_progress: enable tqdm progress bar.
        tolerance: the tolerance under which the model is said to have converged.

    Examples:

        ```python
        monitor = ConvergenceMonitor(tolerance=1e-5)
        for epoch in monitor(model):
            # If model converges, the generator will deplete before the default number
            # of epochs has been reached.
            ...
        ```
    """

    def __init__(
        self,
        number_of_epochs: int = DEFAULT_NUMBER_OF_EPOCHS,
        epochs_per_val: int = DEFAULT_EPOCHS_PER_VAL,
        patience: int = DEFAULT_PATIENCE,
        show_progress: bool = True,
        tolerance: float = 1e-4,
    ):

        self.number_of_epochs = number_of_epochs
        self.tolerance = tolerance
        self.epochs_per_val = epochs_per_val
        self.patience = patience
        self._range = trange if show_progress else range

    @staticmethod
    def _difference_func(new_M, old_M):
        return np.sum((new_M - old_M) ** 2) / np.sum(old_M**2)

    def __call__(self, model: "BaseMF"):
        """A generator that yields epoch numbers."""
        _old_M = model.M
        _model = model
        # mypy does not recognize the union of trange and range as a callable.
        for i in self._range(self.number_of_epochs):  # type: ignore
            yield i  # model is expected to update its M
            should_update = i > self.patience and i % self.epochs_per_val == 0
            if should_update:
                if self._difference_func(_model.M, _old_M) < self.tolerance:
                    break
                _old_M = model.M
