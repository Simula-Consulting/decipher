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
from bokeh.layouts import column, grid, row
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

def filter_people():
    pass

def example_app(source_manager):
    lp = LexisPlot(source_manager)
    lpa = LexisPlotAge(source_manager)
    delta = DeltaScatter(source_manager)
    traj = TrajectoriesPlot(source_manager)
    table = PersonTable(source_manager)
    table.person_table.styles = {"border": "1px solid #e6e6e6", "border-radius": "5px"}
    table.person_table.height = 500
    hist = HistogramPlot(source_manager)

    try:
        args = curdoc().session_context.request.arguments
        pid_list = args.get("pid_list")
    except:
        pid_list = None

    
    if pid_list is not None:
        # filter people
        filter_people()

    lp.figure.x_range = lpa.figure.x_range
    high_risk_person_group = get_filter_element_from_source_manager(
        "High risk - Person", source_manager
    )
    high_risk_exam_group = get_filter_element_from_source_manager(
        "High risk - Exam", source_manager
    )
    vaccine_type = get_filter_element_from_source_manager(
        "Vaccine type", source_manager
    )
    vaccine_group = get_filter_element_from_source_manager(
        "Vaccine age", source_manager
    )
    category_group = get_filter_element_from_source_manager("Region", source_manager)

    filter_grid = grid(
        column(
            row(Div(), Div(text="Active"), Div(text="Invert")),
            high_risk_person_group,
            high_risk_exam_group,
            vaccine_group,
            vaccine_type,
            category_group,
            # get_filter_element_from_source_manager(
            #     "symmetric_difference", source_manager, label="XOR"
            # ),
        )
    )
    # `grid` does not pass kwargs to constructor, so must set attrs here.
    filter_grid.name = "filter_control"
    filter_grid.stylesheets = [
        ":host {grid-template-rows: unset; grid-template-columns: unset;}"
    ]

    for element in (
        lp.figure,
        lpa.figure,
        traj.figure,
        column(
            Div(text="<h1>Data table</h1>"),
            table.person_table,
        ),
        hist.figure,
        delta.figure,
        filter_grid,
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
        "High risk - Person": PersonSimpleFilter(
            source_manager=source_manager,
            person_indices=_at_least_one_high_risk(source_manager.person_source),
        ),
        "High risk - Exam": ExamSimpleFilter(
            source_manager=source_manager,
            exam_indices=[
                i
                for i, state in enumerate(source_manager.exam_source.data["state"])
                if state == 3
            ],
        ),
        "Vaccine age": RangeFilter(source_manager=source_manager, field="vaccine_age"),
        "Vaccine type": CategoricalFilter(
            source_manager=source_manager, field="vaccine_type"
        ),
        "Region": CategoricalFilter(source_manager=source_manager, field="home"),
    }

    # Explicitly make the values a list.
    # dict.values returns a 'view', which will dynamically update, i.e.
    # if we do not take the list, union will have itself in its filters.
    base_filters["symmetric_difference"] = BooleanFilter(
        {name: copy.copy(filter) for name, filter in base_filters.items()},
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
