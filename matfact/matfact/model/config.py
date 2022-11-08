from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import numpy.typing as npt

from matfact import settings
from matfact.model.factorization.utils import convoluted_differences_matrix
from matfact.model.factorization.weights import data_weights


class WeightGetter(ABC):
    """Strategy for getting weights for a given observation matrix.

    If is_identity is True, the weights are assumed to be identity weights, and
    the CMF solver may be used."""

    @abstractmethod
    def __call__(self, X: npt.NDArray) -> npt.NDArray:
        ...

    def __str__(self):
        return f"{self.__class__.__name__}"

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    is_identity: bool = False


class IdentityWeighGetter(WeightGetter):
    def __call__(self, X: npt.NDArray) -> npt.NDArray:
        return X != 0

    is_identity = True


class DataWeightGetter(WeightGetter):
    def __call__(self, X: npt.NDArray) -> npt.NDArray:
        return data_weights(X)


@dataclass(frozen=True)
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

    difference_matrix_getter: Callable[[int], npt.NDArray] = np.identity
    weight_matrix_getter: WeightGetter = field(default_factory=DataWeightGetter)
    minimal_value_matrix_getter: Callable[[tuple[int, int]], npt.NDArray] = np.ones

    def get_short_model_name(self) -> str:
        """Return a short string representing the model.

        The short name consists of three fields, shift, convolution and weights.
        Sample names are scmf, l2mf."""

        # Map possible difference_matrix_getters to string representations
        convolution_mapping = {
            np.identity: "l2",
            convoluted_differences_matrix: "c",
        }

        return (
            "".join(
                (
                    "s" if self.shift_budget else "",
                    convolution_mapping.get(self.difference_matrix_getter, "?"),
                    "" if self.weight_matrix_getter.is_identity else "w",
                )
            )
            + "mf"
        )
