import numpy as np
from numpy.lib.stride_tricks import as_strided


def theta_mle(X, M):
    """Use reconstructed data to estimate the theta parameters used by the prediction
    algorithm.

    Args:
        X: Sparse data matrix
        M: Completed data matrix

    Returns:
        Theta estimate (float)
    """

    O = (X != 0).astype(float)
    return np.sum(O) / (2 * np.square(np.linalg.norm(O * (X - M))))


def initialize_basis(T, r, seed):
    "Random initialization of the basic vectors in the V matrix"

    np.random.seed(seed)
    return np.random.normal(size=(T, r))


def finite_difference_matrix(T):
    "Construct a (T x T) forward difference matrix"

    return np.diag(np.pad(-np.ones(T - 1), (0, 1), "constant")) + np.diag(
        np.ones(T - 1), 1
    )


def laplacian_kernel_matrix(T, gamma=1.0):
    "Construct a (T x T) matrix for convolutional regularization"

    kernel = lambda x: np.exp(-1.0 * gamma * np.abs(x))

    return [kernel(np.arange(T) - i) for i in np.arange(T)]


def reconstruction_mse(true_matrix, observed_matrix, reconstructed_matrix):
    """Compute the reconstruction means-squared error

    Arguments:
     true_matrix: the ground truth dense matrix
     observed_matrix: the observed, censored matrix
     reconstructed_matrix: the reconstruced, dense matrix

    Returns:
     recMSE

    Notes:
     The function uses the observed_matrix to find the observation mask (where the observed_matrix is zero),
     and then finds the norm of the difference at unobserved entries, normalized by the sparseness.
     recMSE = || P_inverse(M - U.V) || / (1-|P|)
     where P is the observation mask, M is the ground truth and U.V is the reconstructed matrix.
    """
    hidden_mask = observed_matrix == 0  # Entries where there is no observation

    return np.linalg.norm(hidden_mask * (true_matrix - reconstructed_matrix)) / np.sum(
        hidden_mask
    )
