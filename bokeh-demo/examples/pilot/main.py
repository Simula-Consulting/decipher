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

import pandas as pd
import numpy as np

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
from bokeh_demo.exam_data import CreatePlottingData
from bokeh_demo.settings import settings
from decipher.data import DataManager


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


def extract_people_from_pids(pid_list, person_df, exams_df):
    return person_df, exams_df

import random

def main():
    # PIDS = get_selected_pids_from_landing_page()
    data_manager = DataManager.read_from_csv(settings.data_paths.screening_data_path, settings.data_paths.dob_data_path)

    person_df, exams_df = data_manager.person_df, data_manager.exams_df
    exams_df = exams_df.sort_values(by="exam_date").reset_index(drop=True)

  
    person_df, exams_df = extract_people_from_pids([], person_df, exams_df)

    # temp fix to replace <NA>
    bool_cols = person_df.select_dtypes(include=['bool']).columns
    person_df[bool_cols] = person_df[bool_cols].astype("float")
    person_df = person_df.fillna(np.nan)

    exams_df["risk"] = exams_df["risk"].astype("float")
    exams_df = exams_df.fillna(np.nan)


    person_df = CreatePlottingData().fit_transform(exams_df)

    person_df["vaccine_age"] = [0] * len(person_df)
    person_df["vaccine_type"] = ["None"] * len(person_df)
    person_df["home"] = [random.choice(list(HomePlaces)) for _ in range(len(person_df))]

    for df in exams_df, person_df:
        category_cols = df.select_dtypes(include=['category']).columns
        for col in category_cols:
            df[col] = df[col].apply(lambda x: x.value)
        #df[category_cols] = df[category_cols].apply(lambda x: x.value)
        df[category_cols] = df[category_cols].astype("str")
        
    source_manager = SourceManager(
        ColumnDataSource(person_df),
        ColumnDataSource(exams_df),
    )
    source_manager.filters = _get_filters(source_manager)

    example_app(source_manager)


# Name is not main when run through bokeh serve, so no __name__ == __main__ guard
main()
