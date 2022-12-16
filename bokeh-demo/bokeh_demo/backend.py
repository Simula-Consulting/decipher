from __future__ import annotations  # Postponed evaluation of types

import itertools
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Sequence, overload

import numpy as np
import numpy.typing as npt
from bokeh.models import CDSView, ColumnDataSource, CustomJS, IndexFilter

from .exam_data import EXAM_RESULT_LOOKUP, ExamResult
from .faker import faker
from .settings import settings

## Time Converter ##


@dataclass
class TimeConverter:
    """Convert between time point index and age."""

    zero_point_age: int = 16
    """Zero point of the measurement data."""
    points_per_year: float = 4
    """Number of time points per year."""

    @overload
    def time_point_to_age(self, time_points: int) -> float:
        ...

    @overload
    def time_point_to_age(self, time_points: Sequence[int]) -> Sequence[float]:
        ...

    def time_point_to_age(self, time_points):
        """Convert time point or points to age."""

        def convert(time):
            return self.zero_point_age + time / self.points_per_year

        try:
            return (convert(time_point) for time_point in time_points)
        except TypeError:  # Only one point
            return convert(time_points)

    @overload
    def age_to_time_point(self, ages: float) -> int:
        ...

    @overload
    def age_to_time_point(self, ages: Sequence[float]) -> Sequence[int]:
        ...

    def age_to_time_point(self, ages):
        """Convert ages to closest time points."""

        def convert(age):
            return round((age - self.zero_point_age) * self.points_per_year)

        try:
            return (convert(age) for age in ages)
        except TypeError:  # Only one point
            return convert(ages)


time_converter = TimeConverter()

## Data structures


def _get_endpoint_indices(history: Sequence[int]) -> tuple[int, int]:
    """Return the first and last index of non-zero entries.

    >>> _get_endpoint_indices((0, 1, 0, 2, 0))
    (1, 3)
    >>> _get_endpoint_indices((0, 1))
    (1, 1)
    """

    def first_nonzero_index(seq):
        return next(i for i, y in enumerate(seq) if y != 0)

    first = first_nonzero_index(history)
    last = len(history) - 1 - first_nonzero_index(reversed(history))
    return first, last


def _calculate_delta(
    probabilities: Sequence[float],
    correct_indices: int,
) -> float:
    """Calculate the delta value from probabilities for different classes."""
    deltas = []
    for estimates, correct in zip(probabilities, correct_indices):
        incorrect_estimates = (*estimates[:correct], *estimates[correct + 1 :])
        # Set default=0 for the edge case that there is only one state, in which
        # case incorrect_estimates is empty.
        deltas.append(max(incorrect_estimates, default=0) - estimates[correct])
    return deltas


@dataclass
class Person:
    index: int
    year_of_birth: float  # Float to allow granular date
    vaccine_age: float | None
    exam_results: Sequence[int]
    detailed_exam_results: Sequence[ExamResult | None]
    predicted_exam_result: int
    prediction_time: int
    prediction_probabilities: Sequence[float]

    def as_source_dict(self):
        """Return a dict representation appropriate for a ColumnDataSource."""
        base_dict = asdict(self)

        # We must have explicit x-values for the plotting
        exam_time_age = list(
            time_converter.time_point_to_age(range(len(self.exam_results)))
        )

        # Delta score of the prediction
        # TODO: we now hack this by using the lists, but in the future a better/more
        # general _calculate_delta should be written
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
        """Return endpoints for the lexis life line"""
        lexis_line_endpoints_person_index = [self.index] * 2

        # The endpoints' indices in the exam result list
        endpoints_indices = _get_endpoint_indices(self.exam_results)
        # Indices to age
        endpoints_age = list(time_converter.time_point_to_age(endpoints_indices))
        endpoints_year = [self.year_of_birth + age for age in endpoints_age]

        # Vaccine life line endpoints
        endpoints_age_vaccine = (
            (self.vaccine_age, endpoints_age[-1])
            if self.vaccine_age is not None
            else ()
        )
        endpoints_year_vaccine = [
            self.year_of_birth + age for age in endpoints_age_vaccine
        ]

        return {
            "lexis_line_endpoints_person_index": lexis_line_endpoints_person_index,
            "lexis_line_endpoints_age": endpoints_age,
            "lexis_line_endpoints_year": endpoints_year,
            "vaccine_line_endpoints_age": endpoints_age_vaccine,
            "vaccine_line_endpoints_year": endpoints_year_vaccine,
        }

    def as_scatter_source_dict(self):
        exam_time_age = list(
            time_converter.time_point_to_age(range(len(self.exam_results)))
        )
        exam_time_year = (self.year_of_birth + age for age in exam_time_age)

        def get_nonzero(seq):
            return [
                element for i, element in enumerate(seq) if self.exam_results[i] != 0
            ]

        return {
            key: get_nonzero(value)
            for key, value in (
                ("age", exam_time_age),
                ("year", exam_time_year),
                ("state", self.exam_results),
                # Used for legend generation
                (
                    "state_label",
                    [settings.label_map[state] for state in self.exam_results],
                ),
                ("person_index", itertools.repeat(self.index, len(self.exam_results))),
            )
        } | {
            "exam_type": [
                exam.type.value for exam in self.detailed_exam_results if exam
            ],
            "exam_result": [
                EXAM_RESULT_LOOKUP[exam.type][exam.result]
                for exam in self.detailed_exam_results
                if exam
            ],
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

            detailed_exam_result = faker.get_fake_detailed_result(exam_result)

            year_of_birth = faker.get_fake_year_of_birth(i)
            vaccine_age = faker.get_fake_vaccine_age()

            people.append(
                Person(
                    index=i,
                    year_of_birth=year_of_birth,
                    vaccine_age=vaccine_age,
                    exam_results=exam_result,
                    detailed_exam_results=detailed_exam_result,
                    predicted_exam_result=prediction_state,
                    prediction_time=prediction_time,
                    prediction_probabilities=self.predicted_probabilities[i],
                )
            )

        return people


def _combine_dicts(dictionaries: Sequence[dict]) -> dict:
    """Combine dictionaries by making lists of observed values.

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
    """Combine dictionaries by making flattened lists of observed values.

    TODO should be combined with the above"""
    dictionary_keys = dictionaries[0].keys()
    assert {key for dic in dictionaries for key in dic.keys()} == set(
        dictionary_keys
    ), "All dictionaries must have the same fields"

    return {
        key: [value for dic in dictionaries for value in dic[key]]
        for key in dictionary_keys
    }


def link_sources(person_source, exam_source):
    def select_person_callback(attr, old, selected_people):
        all_indices = [
            i
            for i, person_index in enumerate(exam_source.data["person_index"])
            if person_index in selected_people
        ]

        exam_source.selected.indices = all_indices
        person_source.selected.indices = selected_people

    def set_group_selected_callback(attr, old, new):
        if new == []:  # Avoid unsetting when hitting a line in scatter plot
            return
        selected_people = list({exam_source.data["person_index"][i] for i in new})
        select_person_callback(None, None, selected_people)

    exam_source.selected.on_change("indices", set_group_selected_callback)
    person_source.selected.on_change("indices", select_person_callback)


class SourceManager:
    def __init__(self, person_source, exam_source):
        self.person_source = person_source
        self.exam_source = exam_source
        link_sources(self.person_source, self.exam_source)

        self.only_selected_view = CDSView(filter=IndexFilter())
        # There is apparently some issues in Bokeh with re-rendering on updating
        # filters. See #7273 in Bokeh
        # https://github.com/bokeh/bokeh/issues/7273
        # The emit seems to resolve this for us, but it is rather hacky.
        self.person_source.selected.js_on_change(
            "indices",
            CustomJS(
                args={"source": self.person_source, "view": self.only_selected_view},
                code="""
            if (source.selected.indices.length){
                view.filter.indices = source.selected.indices;
            } else {
                view.filter.indices = [...Array(source.get_length()).keys()];
            }
            source.change.emit();
            """,
            ),
        )

    @classmethod
    def from_people(cls, people: Sequence[Person]) -> SourceManager:
        return SourceManager(
            cls.source_from_people(people), cls.scatter_source_from_people(people)
        )

    @staticmethod
    def source_from_people(people: Sequence[Person]):
        source_dict = _combine_dicts((person.as_source_dict() for person in people))
        return ColumnDataSource(source_dict)

    @staticmethod
    def scatter_source_from_people(people: Sequence[Person]):
        source_dict = _combine_scatter_dicts(
            [person.as_scatter_source_dict() for person in people]
        )
        return ColumnDataSource(source_dict)
