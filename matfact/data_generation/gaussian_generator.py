import numpy as np


def probability_model(x, theta, dom):
    "The basic probaility model used for data generation"
    return np.exp(-theta * (x - dom) ** 2)


def float_matrix(N, T, r, domain, seed=42):
    """Generate real-valued profiles."""

    np.random.seed(seed)

    centers = np.linspace(70, 170, r)
    x = np.linspace(0, T, T)

    V = np.empty(shape=(T, r))

    for i_r in range(r):
        V[:, i_r] = 1 + 3.0 * np.exp(-5e-4 * (x - centers[i_r]) ** 2)

    U = np.random.gamma(shape=1.0, scale=1.0, size=(N, r))

    M = U @ V.T
    M = domain[0] + (M - np.min(M)) / (np.max(M) - np.min(M)) * (domain[-1] - domain[0])

    return M


def discretise_matrix(M, domain, theta, seed=42):
    """Convert a <float> basis to <int>."""

    np.random.seed(seed)

    d_max = np.max(domain)
    d_min = np.min(domain)

    N, T = M.shape
    Z = domain.shape[0]

    X_float_scaled = d_min + (d_max - d_min) * (M - np.min(M)) / (np.max(M) - np.min(M))

    domain_repeated = np.repeat(domain, N).reshape((N, Z), order="F")

    D = np.empty_like(X_float_scaled)
    for j in range(T):

        column_repeated = np.repeat(X_float_scaled[:, j], 4).reshape((N, 4), order="C")

        pdf = probability_model(column_repeated, theta, domain_repeated)
        cdf = np.cumsum(pdf / np.reshape(np.sum(pdf, axis=1), (N, 1)), axis=1)

        u = np.random.uniform(size=(N, 1))

        D[:, j] = domain[np.argmax(u <= cdf, axis=1)]

    return D
