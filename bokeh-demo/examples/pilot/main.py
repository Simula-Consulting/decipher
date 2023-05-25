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
import json
from dataclasses import dataclass
from enum import Enum

from bokeh.layouts import column, grid, row
from bokeh.models import Div, SymmetricDifferenceFilter, ColumnDataSource
from bokeh.plotting import curdoc

from bokeh_demo.backend import (
    BaseFilter,
    BooleanFilter,
    CategoricalFilter,
    ExamSimpleFilter,
    PersonSimpleFilter,
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
from bokeh_demo.exam_data import ToExam, ExtractPeople


def example_app(source_manager):
    lp = LexisPlot(source_manager)
    lpa = LexisPlotAge(source_manager)
    delta = DeltaScatter(source_manager)
    traj = TrajectoriesPlot(source_manager)
    table = PersonTable(source_manager)
    table.person_table.styles = {"border": "1px solid #e6e6e6", "border-radius": "5px"}
    table.person_table.height = 500
    hist = HistogramPlot(source_manager)

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
                for i, state in enumerate(source_manager.exam_source.data["risk"])
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




def get_selected_pids_from_landing_page():
    """Function to load the selected pids from the landing page."""
    with open(settings.selected_pids_path, "r") as f:
        pid_list: list[int] = json.load(f)
    return pid_list


def main():
    base_path = Path("/Users/thorvald/Documents/Decipher/decipher/matfact/tests/test_datasets")
    dob_path = base_path / "test_dob_data.csv"
    screeingin_path = base_path / "test_screening_data.csv"

    settings.processing.raw_dob_data_path = dob_path
    settings.processing.raw_screening_data_path = screeingin_path
    settings.processing.max_n_females = 100

    screening_data, matfact_pipeline = load_and_process_screening_data(settings.processing.raw_screening_data_path)
    # screening_data = screening_data.dropna(subset="cytMorfologi").astype({"risk": "int"})
    # screening_data = DiagnosisAdder().transform(screening_data).astype({"risk": "int"})
    screening_data["age"] = screening_data["age"] / 365
    exams = ToExam().fit_transform(screening_data).astype({"risk": "int", "bin": int})
    people = ExtractPeople().fit_transform(exams)
    people["home"] = HomePlaces.Other

    source_manager = SourceManager(
        ColumnDataSource(people),
        ColumnDataSource(exams),
    )
    source_manager.filters = _get_filters(source_manager)

    example_app(source_manager)


# Name is not main when run through bokeh serve, so no __name__ == __main__ guard
main()
