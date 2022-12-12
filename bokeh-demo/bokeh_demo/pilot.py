# type: ignore
"""Example Bokeh server app"""
from __future__ import annotations  # Postponed evaluation of types

import argparse
import itertools
import pathlib
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt
import tensorflow as tf
from bokeh.layouts import column, row
from bokeh.models import (
    AllIndices,
    CDSView,
    ColumnDataSource,
    CustomJS,
    CustomJSExpr,
    DataTable,
    Div,
    HoverTool,
    IndexFilter,
    Legend,
    LegendItem,
    Slider,
    TableColumn,
)
from bokeh.models.tickers import FixedTicker
from bokeh.plotting import curdoc, figure
from bokeh.transform import linear_cmap
from matfact.data_generation import Dataset
from matfact.model.config import ModelConfig
from matfact.model.factorization.convergence import ConvergenceMonitor
from matfact.model.matfact import MatFact
from matfact.model.predict.dataset_utils import prediction_data
from matfact.plotting.diagnostic import _calculate_delta
from pydantic import BaseSettings

tf.config.set_visible_devices([], "GPU")

parser = argparse.ArgumentParser()
parser.add_argument("--large-data", action=argparse.BooleanOptionalAction)
args = parser.parse_args()


class Settings(BaseSettings):
    number_of_epochs: int = 100
    label_map: list[str] = ["", "Normal", "Low risk", "High risk", "Cancer"]


settings = Settings()


# Import data
dataset_path = (
    pathlib.Path(__file__).parent.parent
    / "data"
    / ("dataset_large" if args.large_data else "dataset1")
)
dataset = Dataset.from_file(dataset_path)


def _get_enedpoint_indices(history: Sequence[int]) -> tuple[int, int]:
    """Return the first and last index of non-zero entries.

    >>> _get_endpoint_indices((0, 1, 0, 2, 0))
    (1, 3)
    >>> _get_endpoint_indices((0, 1))
    (1, 1)
    """
    first_nonzero_index = lambda seq: next(i for i, y in enumerate(seq) if y != 0)
    first = first_nonzero_index(history)
    last = len(history) - 1 - first_nonzero_index(reversed(history))
    return first, last


@dataclass
class Person:
    index: int
    exam_results: Sequence[int]
    predicted_exam_results: Sequence[int]
    prediction_time: int
    prediction_probabilities: Sequence[float]
    lexis_line_endpoints_age: tuple[int, int]
    # TODO this is sort of unnecessary when we have the endpoits age and date of birth...
    lexis_line_endpoints_year: tuple[int, int]

    def as_source_dict(self):
        """Return a dict representation appropriate for a ColumnDataSource."""
        base_dict = asdict(self)

        # We must have explicit x-values for the plotting
        exam_time_age = [16 + i * 4 for i, _ in enumerate(self.exam_results)]
        return base_dict | {"exam_time_age": exam_time_age}


@dataclass
class PredictionData:
    X_train: npt.NDArray[np.int_]
    X_test: npt.NDArray[np.int_]
    X_test_masked: npt.NDArray[np.int_]
    time_of_prediction: Sequence[int]
    true_state_at_prediction: int
    predicted_probabilities: Sequence[float]
    predicted_states: Sequence[int]

    @classmethod
    def extract_and_predict(
        cls, dataset: Dataset, model: MatFact | None = None
    ) -> PredictionData:
        model = model or MatFact(
            ModelConfig(
                epoch_generator=ConvergenceMonitor(
                    number_of_epochs=settings.number_of_epochs
                ),
            )
        )

        X_train, X_test, _, _ = dataset.get_split_X_M()
        X_test_masked, t_pred_test, x_true_test = prediction_data(X_test)

        model.fit(X_train)
        predicted_probabilities = model.predict_probabilities(X_test, t_pred_test)
        # Could alternatively do matfact._predictor(predicted_probabilites) ...
        predicted_states = model.predict(X_test, t_pred_test)

        return PredictionData(
            X_train=X_train,
            X_test=X_test,
            X_test_masked=X_test_masked,
            time_of_prediction=t_pred_test,
            true_state_at_prediction=x_true_test,
            predicted_probabilities=predicted_probabilities,
            predicted_states=predicted_states,
        )

    def extract_people(self) -> list[Person]:
        # We take the individuals from the test set, not the train set, as
        # it is for these people we have prediction results.
        number_of_individuals, number_of_time_steps = self.X_test.shape

        people = []
        for i in range(number_of_individuals):
            # Generate the predicted exam history by changing the state at exam time
            exam_result = self.X_test[i]
            prediction_time = self.time_of_prediction[i]
            prediction_state = self.predicted_states[i]
            predicted_exam_result = exam_result.copy()
            predicted_exam_result[prediction_time] = prediction_state

            # Find end points of the lexis line
            endpoints_indices = _get_enedpoint_indices(exam_result)
            # TODO: fix
            endpoints_age = [16 + i * 4 for i in endpoints_indices]
            endpoints_year = [1980 + i * 4 for i in endpoints_indices]

            people.append(
                Person(
                    index=i,
                    exam_results=self.X_test[i],
                    predicted_exam_results=predicted_exam_result,
                    prediction_time=prediction_time,
                    prediction_probabilities=self.predicted_probabilities[i],
                    lexis_line_endpoints_age=endpoints_age,
                    lexis_line_endpoints_year=endpoints_year,
                )
            )

        return people


def _combine_dicts(dictionaries: Sequence[dict]) -> dict:
    """Combine dictionaries to one, with the values of the new dict being a list of the values of the old dicts.

    >>> a = {'a': 4}
    >>> b = {'a': 3}
    >>> _combine_dicts((a, b))
    {'a': [4, 3]}
    """
    new_dict = defaultdict(list)
    for dictionary in dictionaries:
        for key, value in dictionary.items():
            new_dict[key].append(value)

    return new_dict


def source_from_people(people: Sequence[Person]):
    source_dict = _combine_dicts((person.as_source_dict() for person in people))
    return ColumnDataSource(source_dict)


def test_plot(source):
    # Bokeh is not very good at failing, so explicitly check that the keys are present
    assert all(key in source.data for key in ("exam_time_age", "exam_results"))

    fig = figure()
    lines = fig.multi_line(
        xs="exam_time_age",
        ys="exam_results",
        source=source,
    )

    curdoc().add_root(fig)


def main():
    prediction_data = PredictionData.extract_and_predict(dataset)
    people = prediction_data.extract_people()
    source = source_from_people(people)
    test_plot(source)


main()
