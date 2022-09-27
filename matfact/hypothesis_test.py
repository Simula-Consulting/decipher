import numpy as np
from hypothesis import assume, given, strategies
from hypothesis.extra.numpy import array_shapes, arrays, from_dtype

from matfact.data_generation.gaussian_generator import discretise_matrix, float_matrix
from matfact.experiments.algorithms.factorization.scmf import (
    _custom_roll,
    _take_per_row_strided,
)


@given(
    strategies.data(),
    strategies.lists(strategies.integers(min_value=-100, max_value=100), min_size=1),
)
def test_float_matrix(data, domain):
    N = data.draw(strategies.integers(min_value=1, max_value=100))
    T = data.draw(strategies.integers(min_value=1, max_value=100))
    r = data.draw(strategies.integers(min_value=1, max_value=min(T, N)))
    M = float_matrix(N, T, r, domain)
    d_min, d_max = np.min(domain), np.max(domain)
    assert not np.isnan(M).any()
    assert np.all(d_min <= M) and np.all(M <= d_max)


@given(
    arrays(
        np.float,
        shape=array_shapes(min_dims=2, max_dims=2),
        elements=from_dtype(np.dtype(np.float), allow_nan=False),
    ),
    strategies.one_of(
        strategies.lists(
            strategies.integers(min_value=-1000, max_value=1000), min_size=2
        ),
        arrays(int, strategies.integers(min_value=1, max_value=7)),
    ),
    strategies.floats(),
)
def test_discretize_matrix(M_array, domain, theta):
    assume(np.min(domain) != np.max(domain))
    discretise_matrix(M_array, domain, theta)


@given(strategies.data())
def test_custom_roll(data):
    """Compare _custom_roll with naive slow rolling"""
    array = data.draw(arrays(np.float, array_shapes(min_dims=2, max_dims=2)))
    assume(not np.isnan(array).any())
    shifts = data.draw(arrays(int, array.shape[0]))
    rolled = _custom_roll(array, shifts)
    for row, rolled_row, shift in zip(array, rolled, shifts):
        assert np.all(rolled_row == np.roll(row, shift))


@given(strategies.data())
def test_take_per_row_strided(data):
    A = data.draw(
        arrays(
            float,
            array_shapes(min_dims=2, max_dims=2, min_side=2),
            elements=strategies.floats(allow_nan=False),
        )
    )
    n_elem = data.draw(  # noqa: F841
        strategies.integers(min_value=0, max_value=A.shape[1] - 1)
    )
    start_idx = data.draw(
        arrays(
            int,
            A.shape[0],
            elements=strategies.integers(
                min_value=0, max_value=A.shape[1] - n_elem - 1
            ),
        )
    )
    strided_A = _take_per_row_strided(A, start_idx, n_elem)
    for i, row in enumerate(strided_A):
        assert np.array_equal(row, A[i, start_idx[i] : start_idx[i] + n_elem])
