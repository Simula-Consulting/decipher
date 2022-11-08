from dataclasses import dataclass, field
from typing import Callable

import numpy.typing as npt

from matfact import settings
from matfact.model.factorization.utils import finite_difference_matrix
from matfact.model.factorization.weights import data_weights


@dataclass
class ModelConfig:
    """Configuration class for the MatFact model."""

    shift_budget: list[int] = field(default_factory=list)

    lambda1: float = 1.0
    lambda2: float = 1.0
    lambda3: float = 1.0

    iter_U: int = 2
    iter_V: int = 2

    learning_rate: float = 0.001
    number_of_states: int = settings.default_number_of_states

    difference_matrix_getter: Callable[[int], npt.NDArray] = finite_difference_matrix
    weight_matrix_getter: Callable[[npt.NDArray], npt.NDArray] = data_weights
