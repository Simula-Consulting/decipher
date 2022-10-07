"""This module demonstrates how to generate synthetic data developed to
esemble the screening data used in the DeCipher project.
"""
import json
import pathlib

import numpy as np
from scipy.stats import betabinom

from matfact.settings import (
    DATASET_PATH,
    default_number_of_states,
    default_observation_probabilities,
    minimum_number_of_observations,
)

from .gaussian_generator import discretise_matrix, float_matrix
from .masking import simulate_mask


def censoring(X, missing=0):
    "Truncate histories to have patterns similar to the real histories"

    t_cens = betabinom.rvs(n=X.shape[1], a=4.57, b=5.74, size=X.shape[0])
    for i, t_end in enumerate(t_cens):
        X[i, t_end:] = missing

    return X


def produce_dataset(
    N,
    T,
    r,
    level,
    memory_length=5,
    missing=0,
    number_of_states: int = default_number_of_states,
    observation_probabilities=None,
    theta=2.5,
    seed=42,
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
            theta: Confidence parameter
            seed: Reference value for pseudo-random generator

    Returns:
            The sparse and original complete data matrices
    """

    M = float_matrix(N=N, T=T, r=r, number_of_states=number_of_states, seed=seed)
    Y = discretise_matrix(M, number_of_states=number_of_states, theta=theta, seed=seed)

    if observation_probabilities is None:
        observation_probabilities = default_observation_probabilities
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
    X = censoring(X, missing=missing)

    valid_rows = np.sum(X != 0, axis=1) >= minimum_number_of_observations

    return X[valid_rows].astype(np.float32), M[valid_rows].astype(np.float32)


class Dataset:
    """Screening dataset container

    This class simplifies generating, loading, and saving datasets.
    Most methods returns the Dataset object, so that it is chainable, as
    >>> Dataset.from_file(some_path).get_X_M()
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
        generation_method="DGD",
        **kwargs,
    ):
        """Generate a Datasert"""
        if generation_method != "DGD":
            raise NotImplementedError("Only DGD generation is implemented.")
        X, M = produce_dataset(N, T, rank, sparsity_level, **kwargs)
        metadata = {
            "rank": rank,
            "sparsity_level": sparsity_level,
            "N": N,
            "T": T,
            "generation_method": generation_method,
            "number_of_states": len(value_range),
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
    rank = 5
    n_rows = 1000
    n_columns = 50
    sparsity_level = 6

    X, M = produce_dataset(N=n_rows, T=n_columns, r=rank, level=sparsity_level)

    dataset_metadata = {
        "rank": rank,
        "sparsity_level": sparsity_level,
        "n_rows": n_rows,
        "n_columns": n_columns,
        "generation_method": "DGD",  # Only one method implemented.
    }

    np.save(DATASET_PATH / "X.npy", X)
    np.save(DATASET_PATH / "M.npy", M)
    with (DATASET_PATH / "dataset_metadata.json").open("w") as metadata_file:
        metadata_file.write(json.dumps(dataset_metadata))


if __name__ == "__main__":
    main()
