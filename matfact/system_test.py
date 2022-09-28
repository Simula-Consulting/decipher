"""Full system integration tests"""
from itertools import product

import numpy as np
import pytest

from examples.example import experiment as experiment1
from examples.example_train_and_log import experiment as experiment2
from matfact.data_generation import Dataset
from matfact.experiments import (
    CMF,
    SCMF,
    WCMF,
    data_weights,
    prediction_data,
    train_and_log,
)


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


def test_train(tmp_path):
    """Run full system test"""
    # Generate some data
    dataset_params = {
        "N": 1000,
        "T": 50,
        "rank": 5,
        "sparsity_level": 6,
    }
    Dataset().generate(**dataset_params).save(tmp_path)

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
    optimization_params = {
        "num_epochs": 10,
        "patience": 2,
    }

    for shift, weight, convolve in product([False, True], repeat=3):
        experiment1(
            hyperparams,
            optimization_params,
            shift,
            weight,
            convolve,
            mlflow_tags=mlflow_tags,
            dataset_path=tmp_path,
        )
        experiment2(
            hyperparams,
            optimization_params,
            shift,
            weight,
            convolve,
            mlflow_tags=mlflow_tags,
            dataset_path=tmp_path,
        )


def test_model_input_not_changed():
    """Test that models do not modify their input arguments.

    Models are expected to leave input variables like X, W, s_budged unmodified
    unless otherwise specified.
    >>> model = WCMF(X, V, W)
    >>> model.run_step()
    model.X should be the same as supplied during initialization.
    """

    # Some arbitrary data size
    sample_size, time_span, rank = 100, 40, 5
    X = np.random.choice(np.arange(5), size=(sample_size, time_span))
    V = np.random.choice(np.arange(5), size=(time_span, rank))
    W = data_weights(X)
    s_budget = np.arange(-5, 5)

    X_initial, W_initial = X.copy(), W.copy()

    cmf = CMF(X, V)
    assert np.array_equal(cmf.X, X_initial)
    cmf.run_step()
    assert np.array_equal(cmf.X, X_initial)

    scmf = WCMF(X, V, W)
    assert np.array_equal(scmf.X, X_initial)
    assert np.array_equal(scmf.W, W_initial)
    scmf.run_step()
    assert np.array_equal(scmf.X, X_initial)
    assert np.array_equal(scmf.W, W_initial)

    scmf = SCMF(X, V, s_budget=s_budget, W=W)
    assert np.array_equal(scmf.X, X_initial)
    assert np.array_equal(scmf.W, W_initial)
    scmf.run_step()
    assert np.array_equal(scmf.X, X_initial)
    assert np.array_equal(scmf.W, W_initial)


def test_model_optional_args():
    """Test that model works when optional arguments are empty."""
    # Some arbitrary data size
    sample_size, time_span, rank = 100, 40, 5
    X = np.random.choice(np.arange(5), size=(sample_size, time_span))
    V = np.random.choice(np.arange(5), size=(time_span, rank))
    s_budget = np.arange(-5, 5)

    cmf = CMF(X, V)
    cmf.run_step()

    scmf = WCMF(X, V)
    scmf.run_step()

    scmf = SCMF(X, V, s_budget=s_budget)
    scmf.run_step()


def test_prediction_data():
    """Test that prediction_data does not alter its input array."""
    methods = ["last_observed"]
    for method in methods:
        rng = np.random.default_rng(42)
        X = rng.integers(0, 2, (4, 10))
        X_passed_to_function = X.copy()
        prediction_data(X_passed_to_function, method)
        assert np.array_equal(X, X_passed_to_function)


def test_value_error_loss_extra_metric():
    """Test that ValueError is raised when loss in extra metric"""
    # Some arbitrary data size
    sample_size, time_span = 100, 40
    X = np.random.choice(np.arange(5), size=(sample_size, time_span))

    with pytest.raises(
        ValueError,
        match=(
            "log_loss True and loss is in extra_metrics. "
            "This is illegal, as it causes name collision!"
        ),
    ):
        train_and_log(
            X,
            X,
            extra_metrics={"loss": lambda x: None},
            log_loss=True,
        )
