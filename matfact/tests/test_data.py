import numpy as np
from hypothesis import assume, given, strategies
from hypothesis.extra.numpy import array_shapes, arrays, from_dtype

from matfact.data_generation import Dataset
from matfact.data_generation.gaussian_generator import discretise_matrix, float_matrix


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


def test_dataset_read_write(tmp_path):
    """Test that datasets are loaded and saved correctly"""
    # Parameters chosen arbitrarily
    dataset_params = {
        "N": 1000,
        "T": 50,
        "rank": 5,
        "sparsity_level": 6,
    }
    Dataset().generate(**dataset_params).save(tmp_path)
    for file in ["X.npy", "M.npy", "dataset_metadata.json"]:
        assert (tmp_path / file).exists()

    imported_dataset = Dataset().load(tmp_path)
    for param in dataset_params:
        assert imported_dataset.metadata[param] == dataset_params[param]

    X, M = imported_dataset.get_X_M()
    N, T = dataset_params["N"], dataset_params["T"]

    # When generating a dataset, some individuals (N) are thrown out due
    # to not having enough non-zero samples.
    assert X.shape[1] == M.shape[1] == T
    assert X.shape[0] <= N and M.shape[0] == X.shape[0]
