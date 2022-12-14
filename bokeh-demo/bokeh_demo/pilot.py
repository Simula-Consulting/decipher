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
from bokeh.transform import factor_cmap, factor_mark, linear_cmap
from matfact.data_generation import Dataset
from matfact.model.config import ModelConfig
from matfact.model.factorization.convergence import ConvergenceMonitor
from matfact.model.matfact import MatFact
from matfact.model.predict.dataset_utils import prediction_data
from matfact.plotting.diagnostic import _calculate_delta
from pydantic import BaseSettings, DirectoryPath

tf.config.set_visible_devices([], "GPU")


class Settings(BaseSettings):
    number_of_epochs: int = 100
    dataset_path: DirectoryPath = pathlib.Path(__file__).parents[1] / "data/dataset1"

    label_map: list[str] = ["", "Normal", "Low risk", "High risk", "Cancer"]
    default_tools: list[str] = [
        "pan",
        "wheel_zoom",
        "box_zoom",
        "save",
        "reset",
        "help",
    ]
    extra_tools: list[str] = ["tap", "lasso_select"]


settings = Settings()


class Faker:
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed=seed)

    def get_fake_year_of_birth(
        self, person_index: int, first_possible: float = 1970, spread: float = 30
    ) -> float:
        """Generate a fake date of birth.

        NB. does not return the same date for a given index. This could
        be fixed by adding some sort of memory."""
        return first_possible + self.rng.random() * spread


faker = Faker()


# Import data
dataset = Dataset.from_file(settings.dataset_path)


def _get_endpoint_indices(history: Sequence[int]) -> tuple[int, int]:
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
    year_of_birth: float  # Float to allow granular date
    exam_results: Sequence[int]
    predicted_exam_result: int
    prediction_time: int
    prediction_probabilities: Sequence[float]

    def as_source_dict(self):
        """Return a dict representation appropriate for a ColumnDataSource."""
        base_dict = asdict(self)

        # We must have explicit x-values for the plotting
        exam_time_age = [16 + i * 4 for i, _ in enumerate(self.exam_results)]

        # Delta score of the prediction
        # TODO: we now hack this by using the lists, but in the future a better/more general _calculate_delta should be written
        delta = _calculate_delta(
            [self.prediction_probabilities],
            [self.exam_results[self.prediction_time] - 1],
        )[0]

        # Generate the predicted states
        predicted_exam_results = self.exam_results.copy()
        predicted_exam_results[self.prediction_time] = self.predicted_exam_result

        return (
            base_dict
            | {
                "exam_time_age": exam_time_age,
                "delta": delta,
                "predicted_exam_results": predicted_exam_results,
            }
            | self.get_lexis_endpoints()
        )

    def get_lexis_endpoints(self):
        """ "Return endpoints for the lexis life line"""
        lexis_line_endpoints_index = [self.index] * 2

        # The endpoints' indices in the exam result list
        endpoints_indices = _get_endpoint_indices(self.exam_results)
        # Indices to age
        endpoints_age = [16 + i / 4 for i in endpoints_indices]
        endpoints_year = [self.year_of_birth + age for age in endpoints_age]

        return {
            "lexis_line_endpoints_index": lexis_line_endpoints_index,
            "lexis_line_endpoints_age": endpoints_age,
            "lexis_line_endpoints_year": endpoints_year,
        }

    def as_scatter_source_dict(self):
        # TODO: fix
        exam_time_age = [16 + i / 4 for i, _ in enumerate(self.exam_results)]
        exam_time_year = (self.year_of_birth + age for age in exam_time_age)

        get_nonzero = lambda seq: [
            element for i, element in enumerate(seq) if self.exam_results[i] != 0
        ]

        return {
            key: get_nonzero(value)
            for key, value in (
                ("age", exam_time_age),
                ("year", exam_time_year),
                ("state", self.exam_results),
                ("person_index", itertools.repeat(self.index, len(self.exam_results))),
            )
        }


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

            year_of_birth = faker.get_fake_year_of_birth(i)

            people.append(
                Person(
                    index=i,
                    year_of_birth=year_of_birth,
                    exam_results=exam_result,
                    predicted_exam_result=prediction_state,
                    prediction_time=prediction_time,
                    prediction_probabilities=self.predicted_probabilities[i],
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


def _combine_scatter_dicts(dictionaries: Sequence[dict]) -> dict:
    """TODO should be combined with the above"""
    dictionary_keys = dictionaries[0].keys()
    assert {key for dic in dictionaries for key in dic.keys()} == set(
        dictionary_keys
    ), "All dictionaries must have the same fields"

    return {
        key: [value for dic in dictionaries for value in dic[key]]
        for key in dictionary_keys
    }


def source_from_people(people: Sequence[Person]):
    source_dict = _combine_dicts((person.as_source_dict() for person in people))
    return ColumnDataSource(source_dict)


def scatter_source_from_people(people: Sequence[Person]):
    source_dict = _combine_scatter_dicts(
        [person.as_scatter_source_dict() for person in people]
    )
    return ColumnDataSource(source_dict)


class LexisPlot:
    _lexis_line_y_key: str = "lexis_line_endpoints_index"
    _lexis_line_x_key: str = "lexis_line_endpoints_age"
    _scatter_y_key: str = "person_index"
    _scatter_x_key: str = "age"

    _markers: list[str] = [None, "square", "circle", "diamond"]
    _marker_colors: list[str] = [None, "blue", "green", "red"]
    _marker_key: str = "state"
    _marker_color_key: str = "state"

    def __init__(self, person_source, scatter_source):
        self.figure = figure(tools=self._get_tools())
        life_line = self.figure.multi_line(
            self._lexis_line_x_key,
            self._lexis_line_y_key,
            source=person_source,
        )
        scatter = self.figure.scatter(
            self._scatter_x_key,
            self._scatter_y_key,
            source=scatter_source,
            color={
                "expr": CustomJSExpr(
                    args={"colors": self._marker_colors},
                    code=f"return this.data.{self._marker_color_key}.map(i => colors[i]);",
                )
            },
        )

    def _get_tools(self):
        return settings.default_tools + settings.extra_tools


class LexisPlotAge(LexisPlot):
    _scatter_y_key = "year"
    _lexis_line_y_key = "lexis_line_endpoints_year"


def get_position_list(array: Sequence) -> Sequence[int]:
    """Given an array, return the position of each element in the sorted list.

    >>> get_position_list([2, 0, 1, 4])
    [2, 0, 1, 3]
    >>> get_position_list([1, 4, 9, 2])
    [0, 2, 3, 2]
    """
    sorted_indices = (i for i, _ in sorted(enumerate(array), key=lambda iv: iv[1]))
    index_map = {n: i for i, n in enumerate(sorted_indices)}
    return [index_map[n] for n in range(len(array))]


class TrajectoriesPlot:
    _exam_color: str = "blue"
    _predicted_exam_color: str = "red"

    def __init__(self, person_source, scatter_source):
        self.figure = figure()

        exam_plot = self.figure.multi_line(
            "exam_time_age",
            "exam_results",
            source=person_source,
            color=self._exam_color,
        )
        predicted_exam_plot = self.figure.multi_line(
            "exam_time_age",
            "predicted_exam_results",
            source=person_source,
            color=self._predicted_exam_color,
        )


class DeltaScatter:
    _delta_scatter_x_key: str = "deltascatter__delta_score_index"
    _delta_scatter_y_key: str = "delta"

    def __init__(self, person_source, scatter_source):
        self.figure = figure()

        # Generate a index list based on delta score
        # TODO: consider guard for overwrite
        person_source.data["deltascatter__delta_score_index"] = get_position_list(
            person_source.data["delta"]
        )

        scatter = self.figure.scatter(
            self._delta_scatter_x_key,
            self._delta_scatter_y_key,
            source=person_source,
        )


def test_plot(person_source, exam_source):
    lp = LexisPlot(person_source, exam_source)
    lpa = LexisPlotAge(person_source, exam_source)
    delta = DeltaScatter(person_source, exam_source)
    traj = TrajectoriesPlot(person_source, exam_source)

    curdoc().add_root(
        row(lp.figure, lpa.figure, delta.figure, traj.figure),
    )


def main():
    prediction_data = PredictionData.extract_and_predict(dataset)
    people = prediction_data.extract_people()
    person_source = source_from_people(people)
    exam_source = scatter_source_from_people(people)
    test_plot(person_source, exam_source)


main()
