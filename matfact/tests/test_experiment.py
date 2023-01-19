import pathlib
from contextlib import contextmanager

import mlflow
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays

from matfact.model import CMF, SCMF, WCMF, data_weights, prediction_data, train_and_log
from matfact.model.config import ModelConfig
from matfact.model.factorization.convergence import ConvergenceMonitor
from matfact.model.logging import (
    MLFlowLogger,
    MLFlowLoggerArtifact,
    MLFlowLoggerDiagnostic,
    MLFlowRunHierarchyException,
    _aggregate_fields,
    dummy_logger_context,
)
from matfact.plotting.diagnostic import _calculate_delta
from matfact.settings import settings


def test_aggregate_fields():
    # Field values must be floats, so define some verbose variables with arbitrary
    # numerical values
    foo = 1.0
    bar = 2.0
    data = [
        {"field1": foo, "field2": foo},
        {"field1": foo, "field2": bar},
    ]
    correct_out = {
        "field1": foo,
        "field2_0": foo,
        "field2_1": bar,
        "field2_mean": np.mean((foo, bar)),
        "field2_std": np.std((foo, bar)),
    }
    out = _aggregate_fields(data)
    assert out == correct_out

    foo_string = "type1"
    bar_string = "type2"
    data = [
        {"field1": foo_string, "field2": foo_string},
        {"field1": foo_string, "field2": bar_string},
    ]
    correct_out = {
        "field1": foo_string,
        "field2_0": foo_string,
        "field2_1": bar_string,
        "field2_mean": float("nan"),
        "field2_std": float("nan"),
    }
    out = _aggregate_fields(data)
    assert set(correct_out) == set(out)
    for field, correct_value in correct_out.items():
        if isinstance(correct_value, float) and np.isnan(correct_value):
            assert np.isnan(out[field])
        else:
            assert out[field] == correct_value


def _artifact_path_from_run(run: mlflow.entities.Run):
    """Return pathlib.Path object of run artifact directory."""
    # When artifact_uri is a file, it is a string prefixed with 'file:', then the
    # string representation of the artifact directory.
    file_prefix = "file:"
    file_string = run.info.artifact_uri
    assert file_string.startswith(file_prefix), "The artifact storage type is not file."
    return pathlib.Path(file_string[len(file_prefix) :])


def test_mlflow_context_hierarchy():
    """Test configurations of nested MLFlowLoggers."""

    with MLFlowLogger(allow_nesting=True):
        pass
    assert mlflow.active_run() is None

    with MLFlowLogger(allow_nesting=False):
        pass
    assert mlflow.active_run() is None

    with pytest.raises(MLFlowRunHierarchyException):
        with MLFlowLogger(allow_nesting=False):
            with MLFlowLogger(allow_nesting=False):
                pass
    assert mlflow.active_run() is None

    with pytest.raises(MLFlowRunHierarchyException):
        with MLFlowLogger(allow_nesting=False):
            with MLFlowLogger(allow_nesting=True):
                with MLFlowLogger(allow_nesting=False):
                    pass
    assert mlflow.active_run() is None

    with MLFlowLogger(allow_nesting=False):
        with MLFlowLogger(allow_nesting=True):
            with MLFlowLogger(allow_nesting=True):
                pass
    assert mlflow.active_run() is None


def test_mlflow_logger(tmp_path):
    """Test the MLFlowLogger context."""
    mlflow.set_tracking_uri(tmp_path)
    artifact_path = tmp_path / "artifacts"
    artifact_path.mkdir()
    mlflow.create_experiment("TestExperiment")

    sample_size, time_span = 100, 40
    U = V = np.random.choice(np.arange(5), size=(sample_size, time_span))
    x_true = x_pred = np.random.choice(np.arange(5), size=(sample_size))
    p_pred = np.random.random(size=(sample_size, 4))
    dummy_output = {
        "params": {},
        "metrics": {},
        "tags": {},
        "meta": {
            "results": {
                "U": U,
                "V": V,
                "x_true": x_true,
                "x_pred": x_pred,
                "p_pred": p_pred,
            }
        },
    }

    # The dummy logger context should not activate a new mlflow run.
    with dummy_logger_context as logger:
        assert mlflow.active_run() is None
        logger(dummy_output)

    # MLFlowLogger should activate an outer run.
    with MLFlowLogger() as logger:
        outer_run = mlflow.active_run()
        assert outer_run is not None
        logger(dummy_output)
        with MLFlowLogger() as inner_logger:
            inner_run = mlflow.active_run()
            assert inner_run is not None
            assert inner_run.data.tags["mlflow.parentRunId"] == outer_run.info.run_id
            inner_logger(dummy_output)
    assert mlflow.active_run() is None

    with MLFlowLoggerArtifact(artifact_path=artifact_path) as logger:
        run_with_artifact = mlflow.active_run()
        logger(dummy_output)
    stored_artifact_path = _artifact_path_from_run(run_with_artifact)
    assert not any(stored_artifact_path.iterdir())  # The directory should be empty.

    with MLFlowLoggerDiagnostic(artifact_path=artifact_path) as logger:
        run_with_artifact = mlflow.active_run()
        logger(dummy_output)
    stored_artifact_path = _artifact_path_from_run(run_with_artifact)
    stored_artifacts = stored_artifact_path.glob("*")
    supposed_to_be_stored = set(
        (
            "basis_.pdf",
            "coefs_.pdf",
            "confusion_.pdf",
            "roc_auc_micro_.pdf",
            "certainty_plot.pdf",
        )
    )
    assert supposed_to_be_stored == set([file.name for file in stored_artifacts])


def test_train_and_log_params():
    """Test that the logger is given all params sent to train_and_log."""
    # Some arbitrary data size
    sample_size, time_span = 100, 40
    X = np.random.choice(np.arange(5), size=(sample_size, time_span))

    hyperparams = {
        "shift_range": list(range(-2, 3)),
        "rank": 5,
        "lambda1": 1,
        "lambda2": 2,
        "lambda3": 3,
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
            for metric in all_metrics:
                assert metric in log_dict["metrics"]

        yield logger

    train_and_log(
        X_test=X,
        X_train=X,
        extra_metrics=extra_metrics,
        logger_context=logger_context(),
        epoch_generator=ConvergenceMonitor(  # Set fewer epochs in order to be faster
            number_of_epochs=10,
            patience=2,
            epochs_per_val=2,
        ),
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
    sample_size, time_span = 100, 40
    X = np.random.choice(np.arange(5), size=(sample_size, time_span))
    W = data_weights(X)
    s_budget = list(range(-5, 5 + 1))

    X_initial, W_initial = X.copy(), W.copy()

    config = ModelConfig(shift_budget=s_budget)

    cmf = CMF(X, config)
    assert np.array_equal(cmf.X, X_initial)
    cmf.run_step()
    assert np.array_equal(cmf.X, X_initial)

    scmf = WCMF(X, config)
    assert np.array_equal(scmf.X, X_initial)
    assert np.array_equal(scmf.W, W_initial)
    scmf.run_step()
    assert np.array_equal(scmf.X, X_initial)
    assert np.array_equal(scmf.W, W_initial)

    scmf = SCMF(X, config)
    assert np.array_equal(scmf.X, X_initial)
    assert np.array_equal(scmf.W, W_initial)
    scmf.run_step()
    assert np.array_equal(scmf.X, X_initial)
    assert np.array_equal(scmf.W, W_initial)


@given(
    arrays(
        int,
        array_shapes(min_dims=2, max_dims=2),
        elements=st.integers(
            min_value=0, max_value=settings.matfact_defaults.number_of_states
        ),
    )
)
def test_prediction_data(X):
    """Test prediction_data."""
    X_passed_to_function = X.copy()
    X_masked, *_ = prediction_data(X_passed_to_function)
    # prediction_data should not alter input
    assert np.array_equal(X, X_passed_to_function)
    # prediction_data should not change the shape
    assert np.array_equal(X.shape, X_masked.shape)


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


def test_data_weights():
    """Test generation of data weights."""
    data_shape = (4, 3)  # Chosen arbitrarily
    observed_data = np.random.randint(
        low=0, high=settings.matfact_defaults.number_of_states, size=data_shape
    )
    weight_per_class = range(settings.matfact_defaults.number_of_states)
    weights = data_weights(observed_data, weight_per_class)

    for i, weight in enumerate(weight_per_class):
        state = i + 1
        assert np.all(weights[observed_data == state] == weight)


@pytest.mark.parametrize(
    "probabilities,correct_index,expected_delta",
    [
        [np.array([[1, 0, 0], [1, 0, 0]]), np.array([0, 0]), np.array([-1, -1])],
        [np.array([[1, 0, 0], [1, 0, 0]]), np.array([1, 1]), np.array([1, 1])],
        [np.array([[0, 1, 0], [0, 0, 1]]), np.array([1, 0]), np.array([-1, 1])],
        [
            np.array([[0.5, 0.5, 0.0], [0.5, 0.0, 0.5]]),
            np.array([0, 1]),
            np.array([0.0, 0.5]),
        ],
    ],
)
def test_delta_score(probabilities, correct_index, expected_delta):
    """Test delta score calculated as expected."""
    assert np.all(_calculate_delta(probabilities, correct_index) == expected_delta)


@pytest.fixture
def factorizer(request):
    sample_size, time_span = 100, 40
    X = np.random.choice(np.arange(5), size=(sample_size, time_span))
    factorizer_class = request.param
    return factorizer_class(X, ModelConfig())


@pytest.mark.parametrize("factorizer", (CMF, WCMF, SCMF), indirect=True)
def test_factorizers_initialized(factorizer):
    """Test that the factorizers initialize their internal matrices."""
    for attr in ("X", "U", "V", "M"):
        assert hasattr(factorizer, attr)


@pytest.mark.parametrize("factorizer", (WCMF, SCMF), indirect=True)
def test_exact_U(factorizer):
    """Test that factorizer initializes U to the exact U."""
    exact_U = factorizer._exactly_solve_U()
    assert np.array_equal(exact_U, factorizer.U)


@pytest.mark.parametrize("factorizer", (WCMF, SCMF), indirect=True)
def test_approx_U(factorizer):
    """Test that factorizer's approximation method approximates the exact solver."""
    exact_U = factorizer._exactly_solve_U()
    approx_U = factorizer._approx_U()
    assert np.allclose(approx_U, exact_U, atol=0.01)  # atol chosen empirically
