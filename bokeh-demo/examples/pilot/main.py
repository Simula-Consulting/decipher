# type: ignore
"""Example Bokeh server app.

The app consist of two main "parts"
  1. The data must be processed and put into sources
  2. The visualization itself
"""

import copy
import json
from enum import Enum

import pandas as pd
from bokeh.layouts import column, grid, row
from bokeh.models import ColumnDataSource, Div, SymmetricDifferenceFilter
from bokeh.plotting import curdoc
from decipher.data import DataManager
from loguru import logger

from bokeh_demo.backend import (
    BaseFilter,
    BooleanFilter,
    ExamSimpleFilter,
    PersonSimpleFilter,
    SourceManager,
)
from bokeh_demo.data_ingestion import (
    CreatePersonSource,
    add_hpv_detailed_information,
    exams_pipeline,
)
from bokeh_demo.frontend import (
    HistogramPlot,
    LexisPlot,
    LexisPlotAge,
    TrajectoriesPlot,
    get_filter_element_from_source_manager,
)
from bokeh_demo.settings import settings


def example_app(source_manager):
    lp = LexisPlot(source_manager)
    lpa = LexisPlotAge(source_manager)
    traj = TrajectoriesPlot(source_manager)
    hist = HistogramPlot(source_manager)

    # Remove delta plot and table as these are related to predictions, which we are not doing
    # delta = DeltaScatter(source_manager)
    # table = PersonTable(source_manager)
    # table.person_table.styles = {"border": "1px solid #e6e6e6", "border-radius": "5px"}
    # table.person_table.height = 500

    lp.figure.x_range = lpa.figure.x_range
    high_risk_person_group = get_filter_element_from_source_manager(
        "High risk - Person", source_manager
    )
    high_risk_exam_group = get_filter_element_from_source_manager(
        "High risk - Exam", source_manager
    )

    # Remove vaccine filters as we do not have vaccine data
    # vaccine_type = get_filter_element_from_source_manager(
    #     "Vaccine type", source_manager
    # )
    # vaccine_group = get_filter_element_from_source_manager(
    #     "Vaccine age", source_manager
    # )
    # category_group = get_filter_element_from_source_manager("Region", source_manager)

    filter_grid = grid(
        column(
            row(Div(), Div(text="Active"), Div(text="Invert")),
            high_risk_person_group,
            high_risk_exam_group,
            # vaccine_group,
            # vaccine_type,
            # category_group,
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
        hist.figure,
        traj.figure,
        filter_grid,
        # Prediction related
        # delta.figure,
        # column(
        #     Div(text="<h1>Data table</h1>"),
        #     table.person_table,
        # ),
    ):
        curdoc().add_root(element)


HIGH_RISK_STATES = {3, 4}
"""Risk levels that are considered high risk."""


def _at_least_one_high_risk(person_source):
    """Return people with at least one high risk"""
    return [
        i
        for i, exam_results in enumerate(person_source.data["exam_results"])
        if not set(exam_results).isdisjoint(HIGH_RISK_STATES)
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
                if state in HIGH_RISK_STATES
            ],
        ),
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


def extract_people_from_pids(
    pid_list: list[int], exams_df: pd.DataFrame
) -> pd.DataFrame:
    return exams_df[exams_df[settings.feature_column_names.PID].isin(pid_list)]


def load_data_manager() -> DataManager:
    try:
        data_manager = DataManager.from_parquet(settings.data_paths.base_path)
    except (FileNotFoundError, ImportError, ValueError) as e:
        logger.exception(e)
        logger.warning("Falling back to .csv loading. This will affect performance.")
        data_manager = DataManager.read_from_csv(
            settings.data_paths.screening_data_path,
            settings.data_paths.dob_data_path,
            read_hpv=True,
        )
    # Add detailed test type and results information to exams_df
    data_manager.exams_df = add_hpv_detailed_information(
        data_manager.exams_df, data_manager.hpv_df
    )
    return data_manager


def main():
    data_manager = load_data_manager()
    exams_df = data_manager.exams_df
    try:
        selected_pids = get_selected_pids_from_landing_page()
        if len(selected_pids) <= 0:
            raise FileNotFoundError("No pids were selected in the landing page.")
        exams_df = extract_people_from_pids(selected_pids, exams_df)
    except FileNotFoundError as e:
        logger.exception(e)
        logger.info("Falling back to displaying all people.")
    exams_df = exams_df.drop(columns=["index"]).reset_index(drop=True)

    exams_df = exams_pipeline.fit_transform(exams_df)
    # Warning!
    # This is not the same DataFrame as `DataManager.person_df`.
    person_df = CreatePersonSource().fit_transform(exams_df)

    # Add column in exams_df referring to the index of a person in person_df, _not_
    # the PID. This is for simpler lookups later
    pid_to_index = {pid: i for i, pid in enumerate(person_df["PID"])}
    exams_df["person_index"] = exams_df["PID"].map(pid_to_index)

    source_manager = SourceManager(
        ColumnDataSource(person_df),
        ColumnDataSource(exams_df),
    )
    source_manager.filters = _get_filters(source_manager)

    example_app(source_manager)


# Name is not main when run through bokeh serve, so no __name__ == __main__ guard
main()
