from contextlib import contextmanager

import numpy as np
import pytest

from matfact.experiments import (
    CMF,
    SCMF,
    WCMF,
    data_weights,
    prediction_data,
    train_and_log,
)


def test_train_and_log_params():
    """Test that the logger is given all params sent to train_and_log."""
    # Some arbitrary data size
    sample_size, time_span = 100, 40
    X = np.random.choice(np.arange(5), size=(sample_size, time_span))

    hyperparams = {
        "shift_range": np.arange(-2, 3),
        "rank": 5,
        "lambda1": 1,
        "lambda2": 2,
        "lambda3": 3,
    }
    optimization_params = {
        "num_epochs": 10,
        "patience": 2,
        "epochs_per_val": 2,
    }
    extra_metrics = {  # Some arbitrary extra metric to log
        "my_metric": lambda model: np.linalg.norm(model.X),
    }
    all_metrics = [*extra_metrics, "loss"]  # We set log_loss=True in train_and_log

    @contextmanager
    def logger_context():
        """Yield a logger that asserts all hyperparams and metrics are present."""

        def logger(log_dict):
            for param in hyperparams:
                # Some params are numpy arrays, so use np.all
                assert np.all(hyperparams[param] == log_dict["params"][param])
            for param in optimization_params:
                assert param not in log_dict["params"]
            for metric in all_metrics:
                assert metric in log_dict["metrics"]

        yield logger

    train_and_log(
        X_test=X,
        X_train=X,
        optimization_params=optimization_params,
        logger_context=logger_context(),
        extra_metrics=extra_metrics,
        log_loss=True,
        **hyperparams
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
