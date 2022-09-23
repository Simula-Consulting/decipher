"""This module demonstrates how to instatiate matrix factorization models
for matrix completion and risk prediction. The example is based on synthetic data
produced in the `datasets` directory.
"""
from typing import Any, Optional

import numpy as np

from .algorithms import CMF, SCMF, WCMF
from .algorithms.utils import (
    finite_difference_matrix,
    initialize_basis,
    laplacian_kernel_matrix,
)


def model_factory(
    X: np.ndarray,
    shift_range: Optional[np.ndarray[Any, int]] = None,
    convolution: bool = False,
    weights: Optional[np.ndarray] = None,
    rank: int = 5,
    seed: int = 42,
    **kwargs
):
    """Initialize and return appropriate model based on arguments.

    kwargs are passed directly to the models.
    """
    if shift_range is None:
        shift_range = np.array([])

    padding = 2 * shift_range.size

    if convolution:
        D = finite_difference_matrix(X.shape[1] + padding)
        K = laplacian_kernel_matrix(X.shape[1] + padding)
    else:
        # None will make the objects generate appropriate identity matrices later.
        D = K = None
    kwargs.update({"D": D, "K": K})

    V = initialize_basis(X.shape[1], rank, seed)

    short_model_name = (
        "".join(
            a if cond else b
            for cond, a, b in [
                (shift_range.size, "s", ""),
                (convolution, "c", "l2"),
                (weights is not None, "w", ""),
            ]
        )
        + "mf"
    )

    if shift_range.size:
        weights = (X != 0).astype(np.float32) if weights is None else weights
        return short_model_name, SCMF(X, V, shift_range, W=weights, **kwargs)
    else:
        if weights is not None:
            return short_model_name, WCMF(X, V, weights, **kwargs)
        else:
            return short_model_name, CMF(X, V, **kwargs)
