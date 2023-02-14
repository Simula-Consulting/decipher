"""This module demonstrates how to generate synthetic data developed to
resemble the screening data used in the DeCipher project.
"""
import json
import pathlib

import numpy as np

from decipher.matfact.settings import settings

from decipher.data_generation.gaussian_synthetic.gaussian_generator import discretise_matrix, float_matrix
from decipher.data_generation.hmm_synthetic.data_generator import simulate_state_histories
from decipher.data_generation.utils import censoring, simulate_mask, time_point_approx, check_matrix


# Data Generation Methods
DGD = "DGD"  # Discrete Gaussian Distribution
HMM = "HMM"  # Hidden Markov Model


def produce_dataset(
    N,
    T,
    r,
    level,
    memory_length=settings.matrix_generation.memory_length,
    missing=0,
    number_of_states=settings.matfact_defaults.number_of_states,
    observation_probabilities=settings.matrix_generation.observation_probabilities,
    theta=settings.matrix_generation.confidence_parameter,
    seed=42,
    censor=True,
    method=DGD,
):
    """Generate a synthetic dataset resembling the real screening data.

    Args:
            N: Number of rows
            T: Number of columns (time points)
            r: Rank of the ground truth factor matrices
            level: Level of missing values
            memory_length: Governs how much influence previous values will
                    have on observing a new value not too far into the future
            missing: How to indicate a missing value
            number_of_states: The number of possible states.
            observation_probabilities: array of observation probabilities
                for the different classes.
            theta: Confidence parameter
            seed: Reference value for pseudo-random generator

    Returns:
            The sparse and original complete data matrices
            The name of the generation method
    """
    # Simulate state histories
    if method == HMM:
        M = simulate_state_histories(
            n_samples=N, points_per_year=time_point_approx(T), seed=42
        )
        N, T = check_matrix(M, N, T, r)
        Y = M
    else:  # if method==DGD:
        M = float_matrix(N=N, T=T, r=r, number_of_states=number_of_states, seed=seed)
        Y = discretise_matrix(
            M, number_of_states=number_of_states, theta=theta, seed=seed
        )


    if number_of_states + 1 != len(observation_probabilities):
        raise ValueError(
            "observation_probabilities must have length one more than the number of states!"  # noqa: E501
        )

    # Simulate a sparse dataset.
    mask = simulate_mask(
        Y,
        observation_proba=observation_probabilities,
        memory_length=memory_length,
        level=level,
        seed=seed,
    )
    X = mask * Y
    X = censoring(X, a=settings.censoring.a, b=settings.censoring.b, missing=missing) if censor else X

    valid_rows = (
        np.sum(X != 0, axis=1)
        >= settings.matrix_generation.minimum_number_of_observations
    )

    return X[valid_rows].astype(np.float32), M[valid_rows].astype(np.float32), method


class Dataset:
    """Screening dataset container

    This class simplifies generating, loading, and saving datasets.

    !!! Note "Chaining"

        Most methods returns the Dataset object, so that it is chainable, as
        ```python
        Dataset.from_file(some_path).get_X_M()
        ```
    """

    def __init__(self, X: np.ndarray, M: np.ndarray, metadata: dict):
        self.X = X
        self.M = M
        self.metadata = metadata

    def __str__(self):
        return (
            "Dataset object of size"
            f" {self.metadata['N']}x{self.metadata['T']} with rank"
            f" {self.metadata['rank']} and sparsity level"
            f" {self.metadata['sparsity_level']}"
        )

    @classmethod
    def from_file(cls, path: pathlib.Path):
        """Load dataset from file"""
        X, M = np.load(path / "X.npy"), np.load(path / "M.npy")
        with (path / "dataset_metadata.json").open("r") as metadata_file:
            metadata = json.load(metadata_file)
        return cls(X, M, metadata)

    def save(self, path: pathlib.Path):
        """Store dataset to file"""
        np.save(path / "X.npy", self.X)
        np.save(path / "M.npy", self.M)
        with (path / "dataset_metadata.json").open("w") as metadata_file:
            metadata_file.write(json.dumps(self.metadata))
        return self

    @classmethod
    def generate(
        cls,
        N,
        T,
        rank,
        sparsity_level,
        produce_dataset_function=produce_dataset,
        number_of_states=settings.matfact_defaults.number_of_states,
        observation_probabilities=settings.matrix_generation.observation_probabilities,
        censor=True,
        method=DGD,
    ):
        """Generate a Dataset

        produce_dataset_function should be a callable with signature
        ```
        Callable(
            N, T, rank, sparsity_level, *, number_of_states, observation_probabilities
        ) -> observed_matrix: ndarray, latent_matrix: ndarray, generation_name: str
        ```
        """
        X, M, generation_name = produce_dataset_function(
            N,
            T,
            rank,
            sparsity_level,
            number_of_states=number_of_states,
            observation_probabilities=observation_probabilities,
            censor=censor,
            method=method,
        )
        number_of_individuals = X.shape[0]
        if number_of_individuals == 0:
            raise RuntimeError("Data generation produced no valid screening data!")

        metadata = {
            "rank": rank,
            "sparsity_level": sparsity_level,
            "N": N,
            "T": T,
            "generation_method": generation_name,
            "number_of_states": number_of_states,
            "observation_probabilities": list(observation_probabilities),
        }

        return cls(X, M, metadata)

    def get_X_M(self):
        """Return the X and M matrix."""
        return self.X, self.M

    def get_split_X_M(self, ratio=0.8):
        """Split dataset into train and test subsets."""
        X, M = self.get_X_M()
        slice_index = int(X.shape[0] * ratio)
        return X[:slice_index], X[slice_index:], M[:slice_index], M[slice_index:]

    def prefixed_metadata(self, prefix="DATASET_"):
        """Return the metadata dict with prefix prepended to keys

        Convenience method used in for example logging"""
        return {prefix + key: value for key, value in self.metadata.items()}


def main():
    rank = settings.matrix_generation.rank
    n_rows = settings.matrix_generation.n_rows
    n_columns = settings.matrix_generation.n_columns
    sparsity_level = settings.matrix_generation.sparsity_level

    X, M = produce_dataset(N=n_rows, T=n_columns, r=rank, level=sparsity_level)

    dataset_metadata = {
        "rank": rank,
        "sparsity_level": sparsity_level,
        "n_rows": n_rows,
        "n_columns": n_columns,
        "generation_method": DGD,  # Only two methods implemented.
    }

    np.save(settings.paths.dataset / "X.npy", X)
    np.save(settings.paths.dataset / "M.npy", M)
    with (settings.paths.dataset / "dataset_metadata.json").open("w") as metadata_file:
        metadata_file.write(json.dumps(dataset_metadata))


if __name__ == "__main__":
    main()
