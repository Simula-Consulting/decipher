import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays, from_dtype

from matfact.config import settings
from matfact.data_generation import Dataset
from matfact.data_generation.gaussian_generator import discretise_matrix, float_matrix

# For very large number of states, the code is very slow. We therefore set some large
# arbitrary max
max_number_of_state = 1000


@given(
    st.data(),
    st.integers(min_value=1, max_value=max_number_of_state),
)
def test_float_matrix(data, number_of_states):
    N = data.draw(st.integers(min_value=1, max_value=100))
    T = data.draw(st.integers(min_value=1, max_value=100))
    r = data.draw(st.integers(min_value=1, max_value=min(T, N)))
    M = float_matrix(N, T, r, number_of_states)
    assert not np.isnan(M).any()

    # Check that all values are within the range.
    # They may be slightly outside due to floating point errors, in that case
    # check that they are close to the domain limits.
    domain_min = 1
    domain_max = number_of_states
    if not (np.all(domain_min <= M) and np.all(M <= domain_max)):
        M_min = np.min(M)
        M_max = np.max(M)
        assert np.isclose(domain_min, M_min) and np.isclose(domain_max, M_max)


@given(
    arrays(
        float,
        shape=array_shapes(min_dims=2, max_dims=2),
        elements=from_dtype(np.dtype(float), allow_nan=False),
    ),
    st.integers(min_value=1, max_value=max_number_of_state),
    st.floats(),
)
def test_discretize_matrix(M_array, number_of_states, theta):
    discretise_matrix(M_array, number_of_states, theta)


def test_dataset_read_write(tmp_path):
    """Test that datasets are loaded and saved correctly"""
    # Parameters chosen arbitrarily
    dataset_params = {
        "N": 1000,
        "T": 50,
        "rank": 5,
        "sparsity_level": 6,
    }
    Dataset.generate(**dataset_params).save(tmp_path)
    for file in ["X.npy", "M.npy", "dataset_metadata.json"]:
        assert (tmp_path / file).exists()

    imported_dataset = Dataset.from_file(tmp_path)
    for param in dataset_params:
        assert imported_dataset.metadata[param] == dataset_params[param]

    X, M = imported_dataset.get_X_M()
    N, T = dataset_params["N"], dataset_params["T"]

    # When generating a dataset, some individuals (N) are thrown out due
    # to not having enough non-zero samples.
    assert X.shape[1] == M.shape[1] == T
    assert X.shape[0] <= N and M.shape[0] == X.shape[0]


def test_dataset_metadata(tmp_path):
    """Test Dataset metadata"""

    # Some arbitrary data
    number_of_individuals = 100
    time_steps = 40
    rank = 5
    sparsity_level = 3
    # Set the number of states to something different from default
    number_of_states = settings.matfact_defaults.number_of_states + 1
    observation_probabilities = np.array(
        (
            *settings.data_generation.observation_probabilities,
            0.4,
        )  # 0.4 chosen arbitrarily
    )
    dataset = Dataset.generate(
        number_of_individuals,
        time_steps,
        rank,
        sparsity_level,
        number_of_states=number_of_states,
        observation_probabilities=observation_probabilities,
    )

    metadata_fields = set(dataset.metadata)
    # Assert the metadata contains the same keys as before
    assert set(dataset.metadata) == metadata_fields

    correct_metadata_subset = {
        "N": number_of_individuals,
        "T": time_steps,
        "rank": rank,
        "sparsity_level": sparsity_level,
        "number_of_states": number_of_states,
    }
    # Assert that the values specified are in the metadata with the correct value
    for key, value in correct_metadata_subset.items():
        assert dataset.metadata[key] == value

    # Assert that a dataset loaded from file has the correct metadata
    dataset.save(tmp_path)
    other_dataset = Dataset.from_file(tmp_path)
    assert set(other_dataset.metadata) == metadata_fields
    for field, value in other_dataset.metadata.items():
        assert value == dataset.metadata[field]

    # Assert that prefixing with no prefix does nothing
    assert set(dataset.prefixed_metadata("")) == metadata_fields


def test_generate_higher_states(monkeypatch):
    """Test Dataset generation with different number of states."""

    # Some arbitrary data
    number_of_individuals = 2000
    time_steps = 100
    rank = 5
    sparsity_level = 1
    # Disable censoring, in order to avoid masking out the states we want to assert.
    monkeypatch.setattr(
        "matfact.data_generation.dataset.censoring", lambda X, missing=0: X
    )
    for i in [-1, 0, 1]:
        # Set the number of states to something different from default
        number_of_states = settings.matfact_defaults.number_of_states + i
        # Disable masking by setting all observation probabilities to one
        observation_probabilities = np.ones(number_of_states + 1)

        dataset = Dataset.generate(
            number_of_individuals,
            time_steps,
            rank,
            sparsity_level,
            number_of_states=number_of_states,
            observation_probabilities=observation_probabilities,
        )
        states_observed = set(dataset.X.flatten())
        states_required = set(range(number_of_states + 1))
        assert states_observed == states_required


DGD_artifact_path = settings.paths.test / "test_artifacts" / "DGD_test.npy"


def _generate_DGD_data() -> Dataset:
    number_of_individuals = 1000
    time_steps = 40
    rank = 5
    sparsity_level = 10
    return Dataset.generate(number_of_individuals, time_steps, rank, sparsity_level)


def test_DGD_generation_snapshot():
    """Test that DGD dataset matches snapshot."""
    dataset = _generate_DGD_data()
    correct_dataset = np.load(DGD_artifact_path)
    assert np.allclose(dataset.X, correct_dataset)
