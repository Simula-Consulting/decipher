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


def train_and_log(
    X_train: np.ndarray,
    X_test: np.ndarray,
    dict_to_log: Optional[dict] = None,
    extra_metrics: Optional[dict[str, Callable[[Type[BaseMF]], float]]] = None,
    log_loss: bool = True,
    nested: bool = False,
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
                "log_loss True and loss is in extra_metrics."
                "This is illegal, as it causes name collision!"
            )
        metrics.append("loss")

    X_test_masked, t_pred, x_true = prediction_data(X_test, "last_observed")
    mlflow.start_run(nested=nested)

    # Create model
    model_name, factoriser = model_factory(X_train, **hyperparams)

    # Fit model
    results = factoriser.matrix_completion(extra_metrics=extra_metrics)

    # Predict
    p_pred = factoriser.predict_probability(X_test_masked, t_pred)
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

    # Logging
    mlflow.log_params(hyperparams)
    mlflow.log_param("model_name", model_name)
    if dict_to_log:
        mlflow.log_params(dict_to_log)

    mlflow.log_metric("matthew_score", score)
    for metric in metrics:
        for epoch, metric_value in zip(results["epochs"], results[metric]):
            mlflow.log_metric(metric, metric_value, step=epoch)
    results["mlflow_run_id"] = mlflow.active_run().info.run_id
    mlflow.end_run()
    return results
