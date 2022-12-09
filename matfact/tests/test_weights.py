import hypothesis.strategies as st
import numpy as np
import tensorflow as tf
from hypothesis import given, settings
from hypothesis.extra.numpy import array_shapes, arrays

from matfact.model.factorization.weights import Projection, propensity_weights


@settings(deadline=None)
@given(
    arrays(
        float,
        shape=array_shapes(min_dims=2, max_dims=2),
        elements=st.integers(min_value=0, max_value=5),
    )
)
def test_propensity_weights(observation_matrix) -> None:
    W_p = propensity_weights(observation_matrix, n_iter=2)
    assert isinstance(W_p, np.ndarray)
    assert W_p.shape == observation_matrix.shape

    W = np.zeros_like(observation_matrix)
    W[observation_matrix != 0] = 1

    assert np.all(W_p >= W)


def generate_array():
    return arrays(
        float,
        shape=array_shapes(min_dims=2, max_dims=2, min_side=10),
        elements=st.floats(min_value=0, max_value=10),
    )


@settings(deadline=None)
@given(
    arr=generate_array(),
    tau=st.floats(min_value=0.1, max_value=30),
    gamma=st.floats(min_value=0.1, max_value=30),
)
def test_projection(arr, tau, gamma) -> None:
    tensor = tf.cast(arr, dtype=tf.float32)
    m, n = tensor.shape
    tau_mn = tau * np.sqrt(m * n)

    constraints = Projection(tau_mn=tau_mn, gamma=gamma)
    projected_tensor = constraints(tensor)

    s = tf.linalg.svd(projected_tensor, compute_uv=False)
    assert sum(s) < tau_mn or np.isclose(sum(s), tau_mn, atol=0.5)
    assert np.all((projected_tensor <= gamma))
