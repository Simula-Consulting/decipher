from typing import Callable, Optional, Type

import numpy as np
from sklearn.metrics import matthews_corrcoef

from matfact.model import CMF, SCMF, WCMF, BaseMF
from matfact.model.config import DataWeightGetter, IdentityWeighGetter, ModelConfig
from matfact.model.factorization.convergence import EpochGenerator
from matfact.model.factorization.utils import convoluted_differences_matrix
from matfact.model.logging import MLFlowLogger
from matfact.model.predict.classification_tree import estimate_probability_thresholds
from matfact.model.predict.dataset_utils import prediction_data


def model_factory(
    X: np.ndarray,
    shift_range: Optional[list[int]] = None,
    use_convolution: bool = False,
    use_weights: bool = True,
    rank: int = 5,
    **kwargs,
):
    """Initialize and return appropriate model based on arguments.

    kwargs are passed the ModelConfig.
    """
    if shift_range is None:
        shift_range = []

    if use_convolution:
        difference_matrix_getter = convoluted_differences_matrix
    else:
        difference_matrix_getter = np.identity

    config = ModelConfig(
        shift_budget=shift_range,
        difference_matrix_getter=difference_matrix_getter,
        weight_matrix_getter=(
            DataWeightGetter() if use_weights else IdentityWeighGetter()
        ),
        rank=rank,
        **kwargs,
    )

    if len(shift_range):
        return SCMF(X, config)
    if config.weight_matrix_getter.is_identity:
        return CMF(X, config)
    else:
        return WCMF(X, config)


def train_and_log(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    epoch_generator: EpochGenerator | None = None,
    dict_to_log: Optional[dict] = None,
    extra_metrics: Optional[dict[str, Callable[[Type[BaseMF]], float]]] = None,
    log_loss: bool = True,
    logger_context=None,
    use_threshold_optimization: bool = True,
    **hyperparams,
):
    """Train model and log in MLFlow.

    Arguments:
        X_train: Training data.
        X_test: Test data
        dict_to_log:  optional dictionary associated with the run, logged with MLFlow.
        extra_metrics: optional dictionary of metrics logged in each epoch of training.
            See `BaseMF.matrix_completion` for more details.
        log_loss: Whether the loss function as function of epoch should be logged
            in MLFlow. Note that this is slow.
        nested: If True, the run is logged as a nested run in MLFlow, useful in for
            example hyperparameter search. Assumes there to exist an active parent run.
        use_threshold_optimization: Use ClassificationTree optimization to find
            thresholds for class selection.
            Can improve results on data skewed towards normal.

    Returns:
        A dictionary of relevant output statistics.


    !!! Note "Cross-validation"

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

    metrics = list(extra_metrics.keys()) if extra_metrics else []
    if log_loss:
        if "loss" in metrics:
            raise ValueError(
                "log_loss True and loss is in extra_metrics. "
                "This is illegal, as it causes name collision!"
            )
        metrics.append("loss")

    with logger_context as logger:

        # Create model
        factoriser = model_factory(X_train, **hyperparams)

        # Fit model
        results = factoriser.matrix_completion(
            extra_metrics=extra_metrics, epoch_generator=epoch_generator
        )

        # Predict
        X_test_masked, t_pred, x_true = prediction_data(X_test)
        p_pred = factoriser.predict_probability(X_test_masked, t_pred)

        mlflow_output: dict = {
            "params": {},
            "metrics": {},
            "tags": {},
            "meta": {},
        }
        if use_threshold_optimization:
            # Find the optimal threshold values
            X_train_masked, t_pred_train, x_true_train = prediction_data(X_train)
            p_pred_train = factoriser.predict_probability(X_train_masked, t_pred_train)
            classification_tree = estimate_probability_thresholds(
                x_true_train,
                p_pred_train,
                [0] * x_true_train.shape[0],
                1,
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
        mlflow_output["params"]["model_name"] = factoriser.config.get_short_model_name()
        if dict_to_log:
            mlflow_output["params"].update(dict_to_log)

        mlflow_output["metrics"]["matthew_score"] = score
        for metric in metrics:
            mlflow_output["metrics"][metric] = results[metric]
        logger(mlflow_output)
    return mlflow_output
