"""This module demonstrates how to instatiate matrix factorization models
for matrix completion and risk prediction. The example is based on synthetic data
produced in the `datasets` directory.
"""
from typing import Any, Callable, Optional, Type

import mlflow
import numpy as np
from sklearn.metrics import matthews_corrcoef

from matfact.experiments import CMF, SCMF, WCMF, BaseMF
from matfact.experiments.algorithms.utils import (
    finite_difference_matrix,
    initialize_basis,
    laplacian_kernel_matrix,
)
from matfact.experiments.predict.clf_tree import estimate_probability_thresholds
from matfact.experiments.simulation.dataset import prediction_data


def model_factory(
    X: np.ndarray,
    shift_range: Optional[np.ndarray[Any, int]] = None,
    convolution: bool = False,
    weights: Optional[np.ndarray] = None,
    rank: int = 5,
    seed: int = 42,
    **kwargs,
):
    """Initialize and return appropriate model based on arguments.

    kwargs are passed directly to the models.
    """
    if shift_range is None:
        shift_range = np.array([])

    padding = 2 * shift_range.size

    if convolution:
        D = finite_difference_matrix(X.shape[1] + padding)
        K = laplacian_kernel_matrix(X.shape[1] + padding)
    else:
        # None will make the objects generate appropriate identity matrices later.
        D = K = None
    kwargs.update({"D": D, "K": K})

    V = initialize_basis(X.shape[1], rank, seed)

    short_model_name = (
        "".join(
            a if cond else b
            for cond, a, b in [
                (shift_range.size, "s", ""),
                (convolution, "c", "l2"),
                (weights is not None, "w", ""),
            ]
        )
        + "mf"
    )

    if shift_range.size:
        weights = (X != 0).astype(np.float32) if weights is None else weights
        return short_model_name, SCMF(X, V, shift_range, W=weights, **kwargs)
    else:
        if weights is not None:
            return short_model_name, WCMF(X, V, weights, **kwargs)
        else:
            return short_model_name, CMF(X, V, **kwargs)


def mlflow_logger(log_data: dict) -> None:
    """Log results dictionary to MLFlow.

    Given a dictionary on the format below, add the run to mlflow.
    Assumes there to be an active MLFlow run!

    Params and metrics should have values that are floats.
    Metric also accepts a list of floats, in which case they are interpreted as
    the metric value as a function of the epochs.
    {
        "params": {"param1": value, "param2": value,},
        "metrics": {"metric1": value, "metric2": [...]},
        "tags": {},
        "meta": {},  # Data not logged to MLFLow
    }"""
    for parameter, value in log_data["params"].items():
        mlflow.log_param(parameter, value)
    for metric, value in log_data["metrics"].items():
        if isinstance(value, list):
            for i in range(len(value)):
                mlflow.log_metric(metric, value[i], step=i)
        else:
            mlflow.log_metric(metric, value)
    mlflow.set_tags(log_data["tags"])


def _mean_and_std(field_name: str, values: list[float] | list[list[float]]) -> dict:
    """Return a dict with mean and standard deviation of the values.

    If the entries of values are lists, use the last element of each, i.e. the mean
    and std at the last epoch.
    """
    if isinstance(values[0], list):
        values = [value[-1] for value in values]
    return {
        f"{field_name}_mean": np.mean(values),
        f"{field_name}_std": np.std(values),
    }


def _store_subruns(field_name: str, values: list[float] | list[list[float]]) -> dict:
    return {f"{field_name}_{i}": value for i, value in enumerate(values)}


def _aggregate_fields(
    data: list[dict],
    aggregate_funcs: list[Callable[[str, list[float] | list[list[float]]], dict]]
    | None = None,
) -> dict:
    """Combine data for fields.

    If all entries has the same value for a given field, the output will have that
    field value combination. For fields with different values for the various
    runs, however, the output will log each value of that field.
    For example, if the field "field2" is foo in the first run and bar in the second,
    the output will have the fields "field1_0": foo, "field1_1": bar.

    Note that if the value is a lsit, it is interpreted as being a log over the epochs,
    and always taken to be unique for each run.

    `aggregate_func` is a function Callable[field_name: str, values: list] that adds
    extra fields for values that are not equal accross runs. By default, mean and
    standard deviation.

    data = [
        {"field1": foo, "field2": foo, ...},
        {"field1": foo, "field2": bar, ...},
    ]

    -> {"field1": foo, "field2_0": foo, "field2_1": bar,}
    """

    # Assume each entry has the same fields
    num_entries = len(data)
    new_data = {}
    if aggregate_funcs is None:
        aggregate_funcs = [_store_subruns, _mean_and_std]
    for field in data[0]:
        values = [data[i][field] for i in range(num_entries)]
        should_separate = (
            isinstance(values[0], list | np.ndarray) or len(set(values)) > 1
        )
        if should_separate:  # Data different, we must aggregate
            for aggregation_function in aggregate_funcs:
                new_data.update(aggregation_function(field, values))
        else:  # All runs have the same data
            new_data[field] = values[0]

    return new_data


def batch_mlflow_logger(log_data: list[dict]) -> None:
    """Combine and log a set of runs.

    Used in for example cross validation training, where all folds should be logged
    as one run.

    Arguments:
    log_data: list of run data. Each entry in log_data should be compatible with the
        format expected by `_mlflow_logger`.
        {
            "params": {"field1": value1,},
            "metrics": {"field2": foo, "field_history": [...],},
            "tags": {},
        }
    """
    new_log = {
        "params": {},
        "metrics": {},
        "tags": {},
    }

    new_log["params"] = _aggregate_fields([data["params"] for data in log_data])
    new_log["metrics"] = _aggregate_fields([data["metrics"] for data in log_data])

    mlflow_logger(new_log)


def train_and_log(
    X_train: np.ndarray,
    X_test: np.ndarray,
    dict_to_log: Optional[dict] = None,
    extra_metrics: Optional[dict[str, Callable[[Type[BaseMF]], float]]] = None,
    log_loss: bool = True,
    nested: bool = False,
    use_threshold_optimization: bool = True,
    optimization_params: Optional[dict[str, Any]] = None,
    **hyperparams,
):
    """Train model and log in MLFlow.

    Params:
    X_train, X_test: Train and test data.
    dict_to_log:  optional dictionary assosiated with the run, logged with MLFlow.
    extra_metrics: opional dictionary of metrics logged in each epoch of training.
        See `BaseMF.matrix_completion` for more details.
    log_loss: Whether the loss function as function of epoch should be logged
        in MLFlow. Note that this is slow.
    nested: If True, the run is logged as a nested run in MLFlow, useful in for
        example hyperparameter search. Assumes there to exist an active parent run.
    use_threshold_optimization: Use ClassificationTree optimization to find thresholds
        for class selection. Can improve results on data skewed towards normal.
    optimization_params: kwargs passed to `BaseMF.matrix_completion`. Example
        {
        "num_epochs": 2000,  # Number of training epochs.
        "patience": 200,  # Number of epochs before considering early termination.
        "epochs_per_val": 5,  # Consider early termination every `epochs_per_val` epoch.
        }

    Returns:
    A dictionary of relevant output statistics.


    Discussion:
    Concerning cross validation: the function accepts a train and test set. In order
    to do for example cross validation hyperparameter search, simply wrap this
    function in cross validation logic.
    In this case, each run will be logged separately.

    In future versions of this package, it is possible that cross validation will
    be supported directly from within this function.
    However, it is not obvious what we should log, as we log for example the
    loss function of each training run.
    Two examples are to log each run separately or logging all folds together.
    """

    metrics = list(extra_metrics.keys()) if extra_metrics else []
    if log_loss:
        if "loss" in metrics:
            raise ValueError(
                "log_loss True and loss is in extra_metrics. "
                "This is illegal, as it causes name collision!"
            )
        metrics.append("loss")
    if optimization_params is None:
        optimization_params = {}

    mlflow.start_run(nested=nested)

    # Create model
    model_name, factoriser = model_factory(X_train, **hyperparams)

    # Fit model
    results = factoriser.matrix_completion(
        extra_metrics=extra_metrics, **optimization_params
    )

    # Predict
    X_test_masked, t_pred, x_true = prediction_data(X_test, "last_observed")
    p_pred = factoriser.predict_probability(X_test_masked, t_pred)

    mlflow_output = {
        "params": {},
        "metrics": {},
        "tags": {},
        "meta": {},
    }
    if use_threshold_optimization:
        # Find the optimal threshold values
        X_train_masked, t_pred_train, x_true_train = prediction_data(
            X_train, "last_observed"
        )
        p_pred_train = factoriser.predict_probability(X_train_masked, t_pred_train)
        classification_tree = estimate_probability_thresholds(
            x_true_train, p_pred_train
        )
        threshold_values = {
            f"classification_tree_{key}": value
            for key, value in classification_tree.get_params().items()
        }
        mlflow_output["params"].update(threshold_values)

        # Use threshold values on the test set
        x_pred = classification_tree.predict(p_pred)
    else:
        # Simply choose the class with the highest probability
        # Class labels are 1-indexed, so add one to the arg index.
        x_pred = 1 + np.argmax(p_pred, axis=1)

    # Score
    score = matthews_corrcoef(x_pred, x_true)
    results.update(
        {
            "score": score,
            "p_pred": p_pred,
            "x_pred": x_pred,
            "x_true": x_true,
        }
    )
    mlflow_output["meta"]["results"] = results

    # Logging
    mlflow_output["params"].update(hyperparams)
    mlflow_output["params"]["model_name"] = model_name
    if dict_to_log:
        mlflow_output["params"].update(dict_to_log)

    mlflow_output["metrics"]["matthew_score"] = score
    for metric in metrics:
        mlflow_output["metrics"][metric] = results[metric]
    mlflow_logger(mlflow_output)
    mlflow_output["meta"]["mlflow_run_id"] = mlflow.active_run().info.run_id
    mlflow.end_run()
    return mlflow_output
