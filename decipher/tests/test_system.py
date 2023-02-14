"""Full system tests.

Should not be included in coverage reporting, as they simply run a bunch of code."""
import importlib
from itertools import product

import pytest

import decipher.matfact.model.factorization.factorizers.mfbase
from examples.example import experiment as experiment1  # type: ignore
from examples.example_train_and_log import experiment as experiment2  # type: ignore
from decipher.data_generation import Dataset
from decipher.matfact.model.factorization.convergence import ConvergenceMonitor
from decipher.matfact.settings import settings


@pytest.mark.skip(
    reason="Consider move to CI. Requires examples to generate dataset, if none exists."
)
def test_all_examples():
    """Run all example scripts."""
    example_path = settings.paths.base / "examples"
    for example in example_path.glob("*.py"):
        importlib.import_module(f"examples.{example.stem}", "matfact").main()


def test_train(tmp_path, monkeypatch):
    """Run full system test"""
    # Generate some data
    dataset_params = {
        "N": 1000,
        "T": 50,
        "rank": 5,
        "sparsity_level": 6,
    }
    Dataset.generate(**dataset_params).save(tmp_path)

    mlflow_tags = {
        "Developer": "Developer Name",
    }
    # Params chosen semi-arbitrarily
    # Number of epcoh is low to make runs fast
    hyperparams = {
        "rank": 5,
        "lambda1": 10,
        "lambda2": 10,
        "lambda3": 100,
    }
    # Set to fewer epochs than default to be faster
    monkeypatch.setattr(
        decipher.matfact.model.factorization.factorizers.mfbase,
        "ConvergenceMonitor",
        lambda: ConvergenceMonitor(number_of_epochs=10, patience=2),
    )

    for shift, weight, convolve in product([False, True], repeat=3):
        experiment1(
            hyperparams,
            shift,
            weight,
            convolve,
            mlflow_tags=mlflow_tags,
            dataset_path=tmp_path,
        )
        experiment2(
            hyperparams,
            shift,
            weight,
            convolve,
            mlflow_tags=mlflow_tags,
            dataset_path=tmp_path,
        )
