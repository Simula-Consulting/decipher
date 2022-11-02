"""This module demonstrates how to instantiate matrix factorization models
for matrix completion and risk prediction. The example is based on synthetic data
produced in the `datasets` directory.
"""
from typing import Any, Callable, Optional, Type

import numpy as np
from sklearn.metrics import matthews_corrcoef

from matfact.config import ModelConfig
from matfact.experiments import CMF, SCMF, WCMF, BaseMF
from matfact.experiments.algorithms.utils import initialize_basis
from matfact.experiments.logging import MLFlowLogger
from matfact.experiments.predict.clf_tree import estimate_probability_thresholds
from matfact.experiments.simulation.dataset import prediction_data


def model_factory(
    X: np.ndarray,
    model_config: ModelConfig,
    **kwargs,
):
    """Initialize and return appropriate model based on arguments.

    kwargs are passed directly to the models.
    """
    V = initialize_basis(X.shape[1], model_config.rank, model_config.seed)

    # short_model_name = (
    #     "".join(
    #         a if cond else b
    #         for cond, a, b in [
    #             (model_config.shift_range.size, "s", ""),
    #             (model_config.convolution, "c", "l2"),
    #             (model_config.weights is not None, "w", ""),
    #         ]
    #     )
    #     + "mf"
    # )
    short_model_name = "TODO"

    if model_config.shift_range.size:
        return short_model_name, SCMF(X, V, model_config, **kwargs)
    else:
        if model_config.weights_getter is not None:
            return short_model_name, WCMF(X, V, model_config, **kwargs)
        else:
            return short_model_name, CMF(X, V, model_config, **kwargs)


def train_and_log(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    model_config: ModelConfig | None = None,
    dict_to_log: Optional[dict] = None,
    extra_metrics: Optional[dict[str, Callable[[Type[BaseMF]], float]]] = None,
    log_loss: bool = True,
    logger_context=None,
    use_threshold_optimization: bool = True,
    optimization_params: Optional[dict[str, Any]] = None,
    **hyperparams,
):
    """Train model and log in MLFlow.

    Params:
    X_train, X_test: Train and test data.
    dict_to_log:  optional dictionary associated with the run, logged with MLFlow.
    extra_metrics: optional dictionary of metrics logged in each epoch of training.
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
    if logger_context is None:
        logger_context = MLFlowLogger()
    if model_config is None:
        model_config = ModelConfig()

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

    with logger_context as logger:

        # Create model
        model_name, factoriser = model_factory(X_train, model_config, **hyperparams)

        # Fit model
        results = factoriser.matrix_completion(
            extra_metrics=extra_metrics, **optimization_params
        )

        # Predict
        X_test_masked, t_pred, x_true = prediction_data(X_test, "last_observed")
        p_pred = factoriser.predict_probability(X_test_masked, t_pred)

        mlflow_output: dict = {
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
        logger(mlflow_output)
    return mlflow_output
