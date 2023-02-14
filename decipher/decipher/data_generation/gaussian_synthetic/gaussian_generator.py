import numpy as np

from decipher.matfact.settings import settings


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


def float_matrix(N, T, r, number_of_states: int, seed=42):
    """Generate real-valued profiles.

    The rank must be such that r <= min(N, T).

    In the unlikely case that the generated matrix M contains the same value
    in all elements, how to map onto the domain is ambiguous. We *define*
    it to be mapped onto the middle value of domain.

    Note that due to floating point errors, the generated matrix may have
    values slightly outside of domain. This is acceptable for our use.
    """

    if N < 1 or T < 1:
        raise ValueError("N and T must be larger than zero.")
    if r > min(N, T):
        raise ValueError("Rank r cannot be larger than either N or T.")

    rnd = np.random.default_rng(seed=seed)

    centre_min, centre_max = settings.gauss_gen.centre_minmax
    centers = np.linspace(centre_min, centre_max, r)
    x = np.linspace(0, T, T)

    k, theta = settings.gauss_gen.scale_factor, settings.gauss_gen.kernel_param
    V = 1 + k * np.exp(-theta * (x[:, None] - centers) ** 2)

    U = rnd.gamma(shape=1.0, scale=1.0, size=(N, r))

    M = U @ V.T

    return _scale_to_domain(M, 1, number_of_states)


def discretise_matrix(M, number_of_states: int, theta, seed=42):
    """Convert a <float> basis to <int>."""

    np.random.seed(seed)
    N, T = M.shape
    domain = np.arange(1, number_of_states + 1)

    X_float_scaled = _scale_to_domain(M, 1, number_of_states)

    domain_repeated = np.repeat(domain, N).reshape((N, number_of_states), order="F")

    D = np.empty_like(X_float_scaled)
    for j in range(T):

        column_repeated = np.repeat(X_float_scaled[:, j], number_of_states).reshape(
            (N, number_of_states), order="C"
        )

        pdf = probability_model(column_repeated, theta, domain_repeated)
        cdf = np.cumsum(pdf / np.reshape(np.sum(pdf, axis=1), (N, 1)), axis=1)

        u = np.random.uniform(size=(N, 1))

        D[:, j] = domain[np.argmax(u <= cdf, axis=1)]

    return D
