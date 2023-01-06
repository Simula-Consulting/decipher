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

import copy
import random
from dataclasses import dataclass
from enum import Enum

import tensorflow as tf
from bokeh.layouts import column, row
from bokeh.models import Div, SymmetricDifferenceFilter
from bokeh.plotting import curdoc
from matfact.data_generation import Dataset
from matfact.model.config import ModelConfig
from matfact.model.factorization.convergence import ConvergenceMonitor
from matfact.model.matfact import MatFact
from matfact.model.predict.dataset_utils import prediction_data

from bokeh_demo.backend import (
    BaseFilter,
    BooleanFilter,
    CategoricalFilter,
    ExamSimpleFilter,
    Person,
    PersonSimpleFilter,
    PredictionData,
    RangeFilter,
    SimpleFilter,
    SourceManager,
)
from bokeh_demo.frontend import (
    DeltaScatter,
    HistogramPlot,
    LexisPlot,
    LexisPlotAge,
    PersonTable,
    TrajectoriesPlot,
    get_filter_element_from_source_manager,
)
from bokeh_demo.settings import settings

tf.config.set_visible_devices([], "GPU")


# Import data
dataset = Dataset.from_file(settings.dataset_path)


def extract_and_predict(
    dataset: Dataset, model: MatFact | None = None
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


def example_app(source_manager):
    lp = LexisPlot(source_manager)
    lpa = LexisPlotAge(source_manager)
    delta = DeltaScatter(source_manager)
    traj = TrajectoriesPlot(source_manager)
    table = PersonTable(source_manager)
    hist = HistogramPlot(source_manager)
    high_risk_person_group = get_filter_element_from_source_manager(
        "high_risk_person", source_manager
    )
    high_risk_exam_group = get_filter_element_from_source_manager(
        "high_risk_exam", source_manager
    )
    high_risk_decoupled_group = get_filter_element_from_source_manager(
        "high_risk_decoupled", source_manager
    )
    vaccine_group = get_filter_element_from_source_manager(
        "vaccine_age", source_manager
    )
    category_group = get_filter_element_from_source_manager("category", source_manager)

    curdoc().theme = "dark_minimal"
    for element in (
        lp.figure,
        lpa.figure,
        traj.figure,
        table.person_table,
        hist.figure,
        delta.figure,
        column(
            Div(text="<h1>Filter controls</h1>"),
            row(Div(text="Active"), Div(text="Invert"), Div(text="Value")),
            high_risk_person_group,
            high_risk_exam_group,
            high_risk_decoupled_group,
            vaccine_group,
            category_group,
            get_filter_element_from_source_manager(
                "symmetric_difference", source_manager, label="XOR"
            ),
        ),
    ):
        curdoc().add_root(element)


def _at_least_one_high_risk(person_source):
    """Return people with at least one high risk"""
    return [
        i
        for i, exam_results in enumerate(person_source.data["exam_results"])
        if 3 in exam_results
    ]


def _get_filters(source_manager: SourceManager) -> dict[str, BaseFilter]:
    base_filters = {
        "high_risk_person": PersonSimpleFilter(
            source_manager=source_manager,
            person_indices=_at_least_one_high_risk(source_manager.person_source),
        ),
        "high_risk_decoupled": SimpleFilter(
            source_manager=source_manager,
            person_indices=_at_least_one_high_risk(source_manager.person_source),
            exam_indices=[
                i
                for i, state in enumerate(source_manager.exam_source.data["state"])
                if state == 3
            ],
        ),
        "high_risk_exam": ExamSimpleFilter(
            source_manager=source_manager,
            exam_indices=[
                i
                for i, state in enumerate(source_manager.exam_source.data["state"])
                if state == 3
            ],
        ),
        "vaccine_age": RangeFilter(source_manager=source_manager, field="vaccine_age"),
        "category": CategoricalFilter(source_manager=source_manager, field="home"),
    }

    # Explicitly make the values a list.
    # dict.values returns a 'view', which will dynamically update, i.e.
    # if we do not take the list, union will have itself in its filters.
    base_filters["symmetric_difference"] = BooleanFilter(
        [copy.copy(filter) for filter in base_filters.values()],
        source_manager,
        bokeh_bool_filter=SymmetricDifferenceFilter,
    )

    return base_filters


# We want to demonstrate categorical data, so extend Person with a custom type having
# the categorical field 'home'.
# We then fake the homes randomly.
class HomePlaces(str, Enum):
    South = "south"
    North = "north"
    East = "east"
    West = "west"
    Other = "other"


@dataclass
class MyPerson(Person):
    home: HomePlaces


def main():
    prediction_data = extract_and_predict(dataset)
    people = prediction_data.extract_people()
    for person in people:
        person.__class__ = MyPerson
        person.home = random.choice(list(HomePlaces)).value
    source_manager = SourceManager.from_people(people)
    source_manager.filters = _get_filters(source_manager)

    example_app(source_manager)


# Name is not main when run through bokeh serve, so no __name__ == __main__ guard
main()
