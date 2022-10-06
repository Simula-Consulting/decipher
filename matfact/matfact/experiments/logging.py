from contextlib import contextmanager
from typing import Callable

import mlflow
import numpy as np

from matfact.plotting import plot_basis, plot_coefs, plot_confusion, plot_roc_curve


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


class MLflowLogger:
    def __init__(self, nested=False):
        self.nested = nested

    def __enter__(self):
        self.run_ = mlflow.start_run(nested=self.nested)
        return self

    def __exit__(self, type, value, traceback):
        return self.run_.__exit__(type, value, traceback)

    def __call__(self, output_dict):
        mlflow_logger(output_dict)


class MLflowBatchLogger(MLflowLogger):
    def __enter__(self):
        self.output = []
        return super().__enter__()

    def __exit__(self, type, value, traceback):
        batch_mlflow_logger(self.output)
        return super().__exit__(type, value, traceback)

    def __call__(self, output_dict):
        self.output.append(output_dict)


class MLflowLoggerArtifact(MLflowLogger):
    def __init__(self, figure_path, nested=False, extra_tags=None):
        super().__init__(nested)
        self.figure_path = figure_path
        self.extra_tags = extra_tags

    def __call__(self, output_dict):
        super().__call__(output_dict)
        if self.extra_tags:
            mlflow.set_tags(self.extra_tags)
        solver_output = output_dict["meta"]["results"]
        plot_coefs(solver_output["U"], self.figure_path)
        plot_basis(solver_output["V"], self.figure_path)
        plot_confusion(
            solver_output["x_true"],
            solver_output["x_pred"],
            self.figure_path,
        )
        plot_roc_curve(
            solver_output["x_true"],
            solver_output["p_pred"],
            self.figure_path,
        )
        mlflow.log_artifacts(self.figure_path)


@contextmanager
def dummy_logger_context(nested=False):
    yield lambda mlflow_output: None
