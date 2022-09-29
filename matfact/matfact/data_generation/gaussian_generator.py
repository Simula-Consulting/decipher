import numpy as np


def probability_model(x, theta, dom):
    "The basic probaility model used for data generation"
    return np.exp(-theta * (x - dom) ** 2)


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

    # Check the edge case that all elements are the same
    if np.all(M == M.flat[0]):
        domain_middle_value = np.min(domain) + (np.max(domain) - np.min(domain)) / 2
        return np.full_like(M, domain_middle_value)

    domain_min, domain_max = np.min(domain), np.max(domain)
    M = domain_min + (M - np.min(M)) / (np.max(M) - np.min(M)) * (
        domain_max - domain_min
    )

    return M


def discretise_matrix(M, domain, theta, seed=42):
    """Convert a <float> basis to <int>."""

    np.random.seed(seed)

    d_max = np.max(domain)
    d_min = np.min(domain)

    N, T = M.shape
    domain = np.array(domain)  # If list is given, convert to numpy array
    Z = len(domain)

    # Check the edge case that all elements are the same
    if np.all(M == M.flat[0]):
        domain_middle_value = d_min + (d_max - d_min) / 2
        X_float_scaled = np.full_like(M, domain_middle_value)
    else:
        X_float_scaled = d_min + (d_max - d_min) * (M - np.min(M)) / (
            np.max(M) - np.min(M)
        )

    domain_repeated = np.repeat(domain, N).reshape((N, Z), order="F")

    D = np.empty_like(X_float_scaled)
    for j in range(T):

        column_repeated = np.repeat(X_float_scaled[:, j], Z).reshape((N, Z), order="C")

        pdf = probability_model(column_repeated, theta, domain_repeated)
        cdf = np.cumsum(pdf / np.reshape(np.sum(pdf, axis=1), (N, 1)), axis=1)

        u = np.random.uniform(size=(N, 1))

        D[:, j] = domain[np.argmax(u <= cdf, axis=1)]

    return D
