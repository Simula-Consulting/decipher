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


def l2_regularizer(X, rank=5, lambda1=1.0, lambda2=1.0, weights=None, seed=42):
    """Matrix factorization with L2 regularization. Weight discrepancy term is optional.

    Args:
        X: Sparse (N x T) data matrix used to estimate factor matrices
        rank: Rank of the factor matrices
        lambda: Regularization coefficients
        weights (optional): Weight matrix (N x T) for the discrepancy term. This matrix
            should have zeros in the same entries as X.

    Returns:
        A CMF/WCMF object with only L2 regularization.
    """

    # Initialize basic vectors
    V = initialize_basis(X.shape[1], rank, seed)

    if weights is None:
        return CMF(X, V, lambda1=lambda1, lambda2=lambda2)

    return WCMF(X, V, W=weights, lambda1=lambda1, lambda2=lambda2)


def convolution(
    X, rank=5, lambda1=1.0, lambda2=1.0, lambda3=1.0, weights=None, seed=42
):
    """Matrix factorization with L2 and convolutional regularization.
    Weighted discrepancy term is optional.
    The convolutional regularization allows
    for more local variability in the reconstructed data.

    Args:
        X: Sparse (N x T) data matrix used to estimate factor matrices
        rank: Rank of the factor matrices
        lambda: Regularization coefficients
        weights (optional): Weight matrix (N x T) for the discrepancy term. This matrix
            should have zeros in the same entries as X.

    Returns:
        A CMF/WCMF object with L2 and convolutional regularization.
    """

    # Initialize basic vectors
    V = initialize_basis(X.shape[1], rank, seed)

    # Construct forward difference (D) and convolutional matrix (K).
    # It is possible to choose another form for the K matrix.
    D = finite_difference_matrix(X.shape[1])
    K = laplacian_kernel_matrix(X.shape[1])

    if weights is None:
        return CMF(X, V, D=D, K=K, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3)

    return WCMF(
        X, V, D=D, K=K, W=weights, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3
    )


def shifted(
    X,
    rank=5,
    s_budget=np.arange(-12, 13),
    lambda1=1.0,
    lambda2=1.0,
    lambda3=1.0,
    weights=None,
    convolution=False,
    seed=42,
):
    """Shifted matrix factorization with L2 and optional convolutional regularization.
    Weighted discrepancy
    term is also optional. The shift

    Note that the shifted models (SCMF) are slower than CMF and WCFM.

    Args:
        X: Sparse (N x T) data matrix used to estimate factor matrices
        rank: Rank of the factor matrices
        s_budget: The range of possible shifts.
        lambda: Regularization coefficients
        weights (optional): Weight matrix (N x T) for the discrepancy term. This matrix
            should have zeros in the same entries as X.
        convolution (bool): If should include convolutional regularisation.

    Returns:
        A CMF/WCMF object with L2 and convolutional regularization.
    """

    V = initialize_basis(X.shape[1], rank, seed)

    D, K = None, None
    if convolution:

        D = finite_difference_matrix(X.shape[1] + 2 * s_budget.size)
        K = laplacian_kernel_matrix(X.shape[1] + 2 * s_budget.size)

    if weights is None:
        return SCMF(
            X,
            V,
            s_budget,
            D=D,
            K=K,
            W=(X != 0).astype(np.float32),
            lambda1=lambda1,
            lambda2=lambda2,
            lambda3=lambda3,
        )

    return SCMF(
        X,
        V,
        s_budget,
        D=D,
        K=K,
        W=weights,
        lambda1=lambda1,
        lambda2=lambda2,
        lambda3=lambda3,
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
