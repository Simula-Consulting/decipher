import hypothesis.strategies as st
import numpy as np
from hypothesis import given, settings
from hypothesis.extra.numpy import array_shapes, arrays

from matfact.model.factorization.weights import propensity_weights


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
