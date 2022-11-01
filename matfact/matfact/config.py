from typing import Callable

import numpy as np
from pydantic import BaseModel, Field, validator


class ModelConfig(BaseModel):
    # Alternatively may set shift_range to np.array([]) directly, as pydantic
    # does a deep copy. However, this gives a deprecation warning, as at some point
    # the truth value is checked.
    # Note that the doc warns that the default_factory argument may change
    # in the future.
    # https://pydantic-docs.helpmanual.io/usage/models/#field-with-dynamic-default-value

    # TODO: We may consider simply having the shift range as a list of ints.
    shift_range: np.ndarray = Field(default_factory=lambda: np.array([]))
    convolution: bool = False
    weights: np.ndarray | None = None  # AKA. W
    rank: int = 5
    seed: int = 42
    differential_matrix: np.ndarray | None = None  # AKA. D
    minimum_values: np.ndarray | None = None  # AKA. J
    convolutional_matrix: np.ndarray | None = None  # AKA. K

    weights_getter: Callable[[np.ndarray], np.ndarray]  # Take in X
    differential_matrix_getter: Callable[[int, int], np.ndarray]  # Take in N, T
    minimum_values_getter: Callable[[int, int], np.ndarray]  # Take in N, T
    convolutional_matrix_getter: Callable[[int, int], np.ndarray]  # Take in N, T

    class Config:
        arbitrary_types_allowed = True  # Allow numpy array types. NB. no validation.
