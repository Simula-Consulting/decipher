from typing import Callable

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field

from matfact import settings

class ParameterConfig(BaseModel):
    lambda1: float = 1.0
    lambda2: float = 1.0
    lambda3: float = 1.0
    learning_rate: float = 0.001
    iter_U: int = 2
    iter_V: int = 2
    rank: int = 5
    number_of_states: int = settings.default_number_of_states


class ModelConfig(BaseModel):
    # Alternatively may set shift_range to np.array([]) directly, as pydantic
    # does a deep copy. However, this gives a deprecation warning, as at some point
    # the truth value is checked.
    # Note that the doc warns that the default_factory argument may change
    # in the future.
    # https://pydantic-docs.helpmanual.io/usage/models/#field-with-dynamic-default-value

    # TODO: We may consider simply having the shift range as a list of ints.
    shift_range: np.ndarray = Field(default_factory=lambda: np.array([]))
    # convolution: bool = False
    # weights: np.ndarray | None = None  # AKA. W
    seed: int = 42
    # differential_matrix: np.ndarray | None = None  # AKA. D
    minimum_values: np.ndarray | None = None  # AKA. J

    weights_getter: Callable[
        [npt.NDArray[np.int_]], npt.NDArray
    ] | None = None  # Take in X
    differential_matrix_getter: Callable[[int], np.ndarray] = np.identity  # Take in N
    # minimum_values_getter: Callable[[int, int], np.ndarray]  # Take in N, T

    class Config:
        arbitrary_types_allowed = True  # Allow numpy array types. NB. no validation.


def finite_difference_matrix(T):
    "Construct a (T x T) forward difference matrix"

    return np.diag(np.pad(-np.ones(T - 1), (0, 1), "constant")) + np.diag(
        np.ones(T - 1), 1
    )


def laplacian_kernel_matrix(T, gamma=1.0):
    "Construct a (T x T) matrix for convolutional regularization"

    def kernel(x):
        return np.exp(-1.0 * gamma * np.abs(x))

    return [kernel(np.arange(T) - i) for i in np.arange(T)]


def kernel_mul_finite_difference(size):
    return finite_difference_matrix(size) @ laplacian_kernel_matrix(size)


# class ConvolutionalModelConfig(ModelConfig):
#     differential_matrix_getter: Callable[
#         [int], np.ndarray
#     ] = kernel_mul_finite_difference
