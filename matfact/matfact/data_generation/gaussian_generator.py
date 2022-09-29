import numpy as np


def probability_model(x, theta, dom):
    "The basic probaility model used for data generation"
    return np.exp(-theta * (x - dom) ** 2)


def _scale_to_domain(X: np.ndarray, domain_min: float, domain_max: float) -> np.ndarray:
    """Scale an array such that all its elements are inside a domain.

    In the case that all elements of X are equal, the scaled array will
    have all elements equal to the middle point of the domain."""

    if np.all(X == X.flat[0]):  # Check if all values are the same.
        # Fill array with middle value of domain.
        return np.full(X.shape, (domain_max + domain_min) / 2)

    X_min, X_max = np.min(X), np.max(X)
    return domain_min + (domain_max - domain_min) * (X - X_min) / (X_max - X_min)


def float_matrix(N, T, r, domain, seed=42):
    """Generate real-valued profiles.

    The rank must be such that r <= min(N, T).

    In the unlikely case that the generated matrix M contains the same value
    in all elements, how to map onto the domain is ambiguous. We *define*
    it to be mapped onto the middle value of domain.

    For very large domains, floating point errors may occur, making
    the output matrix have values outside of matrix.
    An alternative procedure to correct this is the following:
    now, we generate a matrix M, and then map those to the range [0, 1].
    It is possible to instead map it to some integer domain, thus avoiding
    floating point errors when mapping to the target range.

    TODO: fix magic numbers
    """

    if N < 1 or T < 1:
        raise ValueError("N and T must be larger than zero.")
    if r > min(N, T):
        raise ValueError("Rank r cannot be larger than either N or T.")

    np.random.seed(seed)

    centers = np.linspace(70, 170, r)
    x = np.linspace(0, T, T)

    V = np.empty(shape=(T, r))

    for i_r in range(r):
        V[:, i_r] = 1 + 3.0 * np.exp(-5e-4 * (x - centers[i_r]) ** 2)

    U = np.random.gamma(shape=1.0, scale=1.0, size=(N, r))

    M = U @ V.T

    return _scale_to_domain(M, np.min(domain), np.max(domain))


def discretise_matrix(M, domain, theta, seed=42):
    """Convert a <float> basis to <int>."""

    np.random.seed(seed)
    N, T = M.shape
    domain = np.array(domain)  # If list is given, convert to numpy array
    Z = len(domain)

    X_float_scaled = _scale_to_domain(M, np.min(domain), np.max(domain))

    domain_repeated = np.repeat(domain, N).reshape((N, Z), order="F")

    D = np.empty_like(X_float_scaled)
    for j in range(T):

        column_repeated = np.repeat(X_float_scaled[:, j], Z).reshape((N, Z), order="C")

        pdf = probability_model(column_repeated, theta, domain_repeated)
        cdf = np.cumsum(pdf / np.reshape(np.sum(pdf, axis=1), (N, 1)), axis=1)

        u = np.random.uniform(size=(N, 1))

        D[:, j] = domain[np.argmax(u <= cdf, axis=1)]

    return D
