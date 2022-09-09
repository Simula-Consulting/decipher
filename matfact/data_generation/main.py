"""This module demonstrates how to generate synthetic data developed to 
esemble the screening data used in the DeCipher project.
"""
from inspect import classify_class_attrs
import os
import numpy as np
import json

from scipy.stats import betabinom

from .masking import simulate_mask
from .gaussian_generator import float_matrix, discretise_matrix

BASE_PATH = "/Users/thorvald/Documents/Decipher/decipher/matfact/"  # TODO: make generic
BASE_PATH = "./"


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
    value_range=np.arange(1, 5),
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
            value_range: Possible observed values
            theta: Confidence parameter
            seed: Reference value for pseudo-random generator

    Returns:
            The sparse and original complete data matrices
    """

    M = float_matrix(N=N, T=T, r=r, domain=value_range, seed=seed)
    Y = discretise_matrix(M, domain=value_range, theta=theta, seed=seed)

    # Simulate a sparse dataset.
    O = simulate_mask(
        Y,
        observation_proba=np.array([0.01, 0.03, 0.08, 0.12, 0.04]),
        memory_length=memory_length,
        level=level,
        seed=seed,
    )
    X = O * Y
    X = censoring(X, missing=missing)

    valid_rows = np.sum(X != 0, axis=1) > 2

    return X[valid_rows].astype(np.float32), M[valid_rows].astype(np.float32)


class Dataset:
    def __init__(self):
        self.data_loaded = False
        self.metadata = {
            "rank": None,
            "n_rows": None,
            "n_columns": None,
            "sparsity_level": None,
        }

    def load(self, path):
        """Load dataset from file"""
        assert not self.data_loaded, "Data is already loaded!"

        self.X, self.M = np.load(path + "/X.npy"), np.load(path + "/M.npy")
        with open(path + "/dataset_metadata.json", "r") as metadata_file:
            self.metadata.update(json.load(metadata_file))
        self.data_loaded = True
        return self

    def save(self, path):
        """Store dataset to file"""
        assert self.data_loaded, "No data is loaded or generated!"
        np.save(f"{path}/X.npy", self.X)
        np.save(f"{path}/M.npy", self.M)
        with open(f"{path}/dataset_metadata.json", "w") as metadata_file:
            metadata_file.write(json.dumps(self.metadata))
        return self

    def generate(
        self,
        N,
        T,
        r,
        level,
        memory_length=5,
        missing=0,
        value_range=np.arange(1, 5),
        theta=2.5,
        seed=42,
        generation_method="DGD",
    ):
        assert not self.data_loaded, "Data is already loaded!"
        if generation_method != "DGD":
            raise NotImplementedError("Only DGD generation is implemented.")
        self.X, self.M = produce_dataset(
            N,
            T,
            r,
            level,
            memory_length=memory_length,
            missing=missing,
            value_range=value_range,
            theta=theta,
            seed=seed,
        )
        self.metadata = {
            "rank": r,
            "sparsity_level": level,
            "N": N,
            "T": T,
            "generation_method": generation_method,
        }
        self.data_loaded = True
        return self

    def get_X_M(self):
        assert self.data_loaded
        return self.X, self.M

    def get_split_X_M(self):
        X, M = self.get_X_M()
        # TODO: Keep the hardcoded slicing used in original
        # code until we are sure that this hardcoding is not
        # expected anywher else
        # After that we should split using some optional ratio
        return X[:800], X[-200:], M[:800], M[-200:]

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

    np.save(f"{BASE_PATH}/datasets/X.npy", X)
    np.save(f"{BASE_PATH}/datasets/M.npy", M)
    with open(f"{BASE_PATH}/datasets/dataset_metadata.json", "w") as metadata_file:
        metadata_file.write(json.dumps(dataset_metadata))


if __name__ == "__main__":
    main()
