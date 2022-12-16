# type: ignore
"""Example Bokeh server app.

The app consist of two main "parts"
  1. The data must be processed and put into sources
  2. The visualization itself

For part 1, the most important classes are `PredictionData` and `Person`.
`PredictionData` is responsible for reading in the data and then constructing
`Person` instances from it.
The `Person` class is responsible for generating the source objects to be used
by the visualization.
"""
from __future__ import annotations  # Postponed evaluation of types

import itertools
import pathlib
from collections import defaultdict
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Sequence, overload

import numpy as np
import numpy.typing as npt
import tensorflow as tf
from bokeh.layouts import row
from bokeh.models import (
    CDSView,
    Circle,
    ColumnDataSource,
    CustomJS,
    CustomJSExpr,
    CustomJSHover,
    DataTable,
    HoverTool,
    IndexFilter,
    Label,
    Legend,
    LegendItem,
    TableColumn,
)
from bokeh.models.tickers import FixedTicker
from bokeh.plotting import curdoc, figure
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
        "examine",  # For debugging
    ]
    extra_tools: list[str] = ["tap", "lasso_select"]


settings = Settings()


class ExamTypes(str, Enum):
    Cytology = "cytology"
    Histology = "histology"


@dataclass
class ExamResult:
    type: ExamTypes
    result: int  # Must be looked up


EXAM_RESULT_LOOKUP = {
    ExamTypes.Cytology: [
        "CytDiagnosis0",
        "CytDiagnosis1",
        "CytDiagnosis2",
    ],
    ExamTypes.Histology: [
        "HistDiagnosis0",
        "HistDiagnosis1",
        "HistDiagnosis2",
        "HistDiagnosis3",
        "HistDiagnosis4",
    ],
}

# Mapping from diagnosis to coarse state
EXAM_RESULT_MAPPING = {
    ExamTypes.Cytology: [
        1,
        2,
        3,
    ],
    ExamTypes.Histology: [
        1,
        1,
        2,
        3,
        4,
    ],
}


def get_inverse_mapping() -> dict[int, list[ExamResult]]:
    possible_diagnosis = defaultdict(list)
    for type, states in EXAM_RESULT_MAPPING.items():
        for diagnosis_index, coarse_state in enumerate(states):
            possible_diagnosis[coarse_state].append(ExamResult(type, diagnosis_index))
    return dict(possible_diagnosis)  # We want KeyError for unknown states


class Faker:
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed=seed)
        self.coarse_state_to_exam_result = get_inverse_mapping()

    def get_fake_year_of_birth(
        self, person_index: int, first_possible: float = 1970, spread: float = 30
    ) -> float:
        """Generate a fake date of birth.

        NB. does not return the same date for a given index. This could
        be fixed by adding some sort of memory."""
        return first_possible + self.rng.random() * spread

    def get_fake_vaccine_age(
        self,
        vaccine_start_year: float = 12.0,
        vaccine_spread: float = 10,
        vaccine_prob: float = 0.3,
    ) -> None | float:
        # Beta 2, 5 is centered around 0.2 with a steep falloff.
        return (
            vaccine_start_year + vaccine_spread * self.rng.beta(2, 5)
            if self.rng.random() < vaccine_prob
            else None
        )

    def get_fake_detailed_result(
        self, coarse_exam_result: Sequence[int]
    ) -> Sequence[ExamResult | None]:
        return [
            self.rng.choice(self.coarse_state_to_exam_result[state]) if state else None
            for state in coarse_exam_result
        ]


faker = Faker()


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


# Import data
dataset = Dataset.from_file(settings.dataset_path)


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


def source_from_people(people: Sequence[Person]):
    source_dict = _combine_dicts((person.as_source_dict() for person in people))
    return ColumnDataSource(source_dict)


def scatter_source_from_people(people: Sequence[Person]):
    source_dict = _combine_scatter_dicts(
        [person.as_scatter_source_dict() for person in people]
    )
    return ColumnDataSource(source_dict)


class ToolsMixin:
    def _get_tools(self):
        return settings.default_tools + settings.extra_tools


class LexisPlot(ToolsMixin):
    _title: str = "Lexis plot"
    _x_label: str = "Age"
    _y_label: str = "Individual #"

    _lexis_line_y_key: str = "lexis_line_endpoints_person_index"
    _lexis_line_x_key: str = "lexis_line_endpoints_age"
    _vaccine_line_x_key: str = "vaccine_line_endpoints_age"
    _vaccine_line_y_key: str = "lexis_line_endpoints_person_index"
    _scatter_y_key: str = "person_index"
    _scatter_x_key: str = "age"

    _marker_key: str = "state"
    _marker_color_key: str = "state"

    # TODO: move to config class or settings
    _markers: list[str] = [None, "square", "circle", "diamond"]
    _marker_colors: list[str] = [None, "blue", "green", "red"]
    _vaccine_line_width: int = 3
    _vaccine_line_color: str = "tan"

    def __init__(self, person_source, scatter_source):
        self.figure = figure(
            title=self._title,
            x_axis_label=self._x_label,
            y_axis_label=self._y_label,
            tools=self._get_tools(),
        )
        self.life_line = self.figure.multi_line(
            self._lexis_line_x_key,
            self._lexis_line_y_key,
            source=person_source,
        )
        self.vaccine_line = self.figure.multi_line(
            self._vaccine_line_x_key,
            self._vaccine_line_y_key,
            source=person_source,
            line_width=self._vaccine_line_width,
            color=self._vaccine_line_color,
        )

        # Legend
        # TODO: Make more general by using mixin
        # The legend layout code must come before the scatter renderer,
        # which will add new items to the legend.
        self.figure.add_layout(
            Legend(
                items=[
                    LegendItem(label="Vaccine", renderers=[self.vaccine_line], index=0)
                ],
                orientation="horizontal",
            ),
            "above",
        )
        self.scatter = self.figure.circle(
            self._scatter_x_key,
            self._scatter_y_key,
            source=scatter_source,
            color={
                "expr": CustomJSExpr(
                    args={"colors": self._marker_colors},
                    code=f"return this.data.{self._marker_color_key}.map(i => colors[i]);",  # noqa: E501
                )
            },
            legend_group="state_label",
        )

        # Tooltip for detailed exam data
        hover_tool = HoverTool(
            tooltips=[("Type", "@exam_type"), ("Result", "@exam_result")],
            renderers=[self.scatter],
        )
        self.figure.add_tools(hover_tool)
        self.scatter.hover_glyph = Circle(x="x", y="y", line_width=10, line_color="red")


class LexisPlotAge(LexisPlot):
    _y_label: str = "Year"
    _scatter_y_key = "year"
    _lexis_line_y_key = "lexis_line_endpoints_year"
    _vaccine_line_y_key: str = "vaccine_line_endpoints_year"


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


class TrajectoriesPlot(ToolsMixin):
    _exam_color: str = "blue"
    _predicted_exam_color: str = "red"

    def __init__(self, person_source, scatter_source):
        self.figure = figure(x_axis_label="Age", tools=self._get_tools())

        # In order to totally deactivate the lines that are not selected
        # we add a filter which only shows the selected lines.
        # This could be done with a IndexFilter and a callback,
        # but a JS filter is faster.

        self.only_selected_view = CDSView(filter=IndexFilter())
        # There is apparently some issues in Bokeh with re-rendering on updating
        # filters. See #7273 in Bokeh
        # https://github.com/bokeh/bokeh/issues/7273
        # The emit seems to resolve this for us, but it is rather hacky.
        person_source.selected.js_on_change(
            "indices",
            CustomJS(
                args={"source": person_source, "view": self.only_selected_view},
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

        self.exam_plot = self.figure.multi_line(
            "exam_time_age",
            "exam_results",
            source=person_source,
            view=self.only_selected_view,
            color=self._exam_color,
            legend_label="Actual observation",
        )
        self.predicted_exam_plot = self.figure.multi_line(
            "exam_time_age",
            "predicted_exam_results",
            source=person_source,
            view=self.only_selected_view,
            color=self._predicted_exam_color,
            legend_label="Predicted observation",
        )

        # Simple tooltip
        list_formatter = CustomJSHover(
            code="""
        return `[${value.map(n => n.toFixed(2)).join(', ')}]`
        """
        )
        hover_tool = HoverTool(
            tooltips=[
                ("Id", "$index"),
                ("Vaccine", "@vaccine_age{0.0}"),
                ("Probabilities", "@prediction_probabilities{custom}"),
            ],
            formatters={"@prediction_probabilities": list_formatter},
        )
        self.figure.add_tools(hover_tool)

        # Set y-ticks to state names
        self.figure.yaxis.ticker = FixedTicker(
            ticks=list(range(len(settings.label_map)))
        )
        self.figure.yaxis.major_label_overrides = dict(enumerate(settings.label_map))


class DeltaScatter(ToolsMixin):
    _delta_scatter_x_key: str = "deltascatter__delta_score_index"
    _delta_scatter_y_key: str = "delta"

    def __init__(self, person_source, scatter_source):
        self.figure = figure(
            x_axis_label="Individual",
            y_axis_label="Delta score (lower better)",
            tools=self._get_tools(),
        )

        # Generate a index list based on delta score
        # TODO: consider guard for overwrite
        person_source.data["deltascatter__delta_score_index"] = get_position_list(
            person_source.data["delta"]
        )

        self.scatter = self.figure.scatter(
            self._delta_scatter_x_key,
            self._delta_scatter_y_key,
            source=person_source,
        )


class PersonTable:
    def __init__(self, person_source, scatter_source):
        # Add column for correct state and prediction discrepancy
        exam_results = person_source.data["exam_results"]
        exam_times = person_source.data["prediction_time"]
        true_state_at_prediction = [
            exam_result[exam_time]
            for exam_result, exam_time in zip(exam_results, exam_times)
        ]
        prediction_discrepancy = [
            true - predicted
            for true, predicted in zip(
                true_state_at_prediction, person_source.data["predicted_exam_result"]
            )
        ]

        person_source.data["persontable__true_state"] = true_state_at_prediction
        person_source.data["persontable__discrepancy"] = prediction_discrepancy

        self.person_table = DataTable(
            source=person_source,
            columns=[
                TableColumn(title="Delta score", field="delta"),
                TableColumn(title="Predicted state", field="predicted_exam_result"),
                TableColumn(title="Correct state", field="persontable__true_state"),
                TableColumn(
                    title="Prediction discrepancy", field="persontable__discrepancy"
                ),
            ],
            styles={
                "border": "1px solid black",
                "margin-right": "40px",
            },
        )


class LabelSelectedMixin:
    def add_label(self):
        self.label = Label(
            x=10,
            y=410,
            x_units="screen",
            y_units="screen",
            text=self._get_label_text(),
            text_font_size="12px",
            border_line_color="black",
            border_line_alpha=1.0,
            background_fill_color="white",
            background_fill_alpha=1.0,
        )
        self.figure.add_layout(self.label)

    def _get_age_at_exam(self, selected_indices):
        return (
            [
                age
                for age, state in zip(
                    self.person_source.data["exam_time_age"][i],
                    self.person_source.data["exam_results"][i],
                )
                if state != 0
            ]
            for i in selected_indices
        )

    @staticmethod
    def _compute_average_screening_interval(nested_age_at_exam):
        screening_intervals = []
        for x in nested_age_at_exam:
            screening_intervals += np.diff(x).tolist()
        # Convert to months
        return np.mean(screening_intervals) * 12

    def _get_label_text(self, selected_indices=None):
        selected_indices = selected_indices or range(self._number_of_individuals)
        n_vaccines = sum(
            self.person_source.data["vaccine_age"][i] is not None
            for i in selected_indices
        )

        nested_age_at_exam = self._get_age_at_exam(selected_indices)
        average_screening_interval = self._compute_average_screening_interval(
            nested_age_at_exam
        )
        return (
            f" Individuals selected: {len(selected_indices)} \n"
            f" Individuals with vaccinations: {n_vaccines} \n"
            f" Average screening interval: ~{round(average_screening_interval, 2)} months"  # noqa: E501
        )

    def get_update_label_callback(self):
        def update_label_callback(attr, old, new):
            new = new if len(new) else list(range(self._number_of_individuals))
            self.label.text = self._get_label_text(new)

        return update_label_callback


class HistogramPlot(ToolsMixin, LabelSelectedMixin):
    def __init__(self, person_source, exam_source):
        self.person_source = person_source
        self.exam_source = exam_source
        self._number_of_individuals = len(self.person_source.data["index"])

        self.figure = figure(tools=self._get_tools())
        self.quad = self.figure.quad(
            top=self.compute_histogram_data(),
            bottom=0,
            left=np.arange(0, 4) + 0.5,
            right=np.arange(1, 5) + 0.5,
            fill_color="navy",
            line_color="white",
            alpha=0.5,
            name="quad",
        )

        self.person_source.selected.on_change(
            "indices", self.get_update_histogram_callback()
        )

        # Add label from LabelSelectedMixin
        self.add_label()
        self.person_source.selected.on_change(
            "indices", self.get_update_label_callback()
        )

        self._set_properties()

    def _set_properties(self):
        properties = {
            "y_range": {"start": 0},
            "xaxis": {
                "axis_label": "State",
                "ticker": list(range(len(settings.label_map))),
                "major_label_overrides": dict(enumerate(settings.label_map)),
            },
            "yaxis": {"axis_label": "Count"},
            "grid": {"grid_line_color": "white"},
        }

        for module, module_options in properties.items():
            for option, value in module_options.items():
                setattr(getattr(self.figure, module), option, value)

    def compute_histogram_data(self, selected_indices=None):
        selected_indices = selected_indices or range(self._number_of_individuals)
        state_occurrences = self._count_state_occurrences(
            [
                [
                    yi
                    for i in selected_indices
                    for yi in self.person_source.data["exam_results"][i]
                    if yi != 0
                ]
            ]
        )
        return [
            value for _, value in sorted(state_occurrences.items(), key=lambda x: x[0])
        ]

    @staticmethod
    def _count_state_occurrences(nested_list_of_states):
        out = {1: 0, 2: 0, 3: 0, 4: 0}
        for list_of_states in nested_list_of_states:
            for state in list_of_states:
                out[state] += 1
        return out

    def get_update_histogram_callback(self):
        def update_histogram(attr, old, new):
            new = new if len(new) else list(range(self._number_of_individuals))

            self.quad.data_source.data["top"] = self.compute_histogram_data(new)

        return update_histogram


def test_plot(person_source, exam_source):
    lp = LexisPlot(person_source, exam_source)
    lpa = LexisPlotAge(person_source, exam_source)
    delta = DeltaScatter(person_source, exam_source)
    traj = TrajectoriesPlot(person_source, exam_source)
    table = PersonTable(person_source, exam_source)
    hist = HistogramPlot(person_source, exam_source)

    curdoc().add_root(
        row(
            hist.figure,
            lp.figure,
            lpa.figure,
            delta.figure,
            traj.figure,
            table.person_table,
        ),
    )


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


def main():
    prediction_data = PredictionData.extract_and_predict(dataset)
    people = prediction_data.extract_people()
    person_source = source_from_people(people)
    exam_source = scatter_source_from_people(people)
    link_sources(person_source, exam_source)
    test_plot(person_source, exam_source)


main()
