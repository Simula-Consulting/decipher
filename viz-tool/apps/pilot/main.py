"""Example Bokeh server app.

The app consist of two main "parts"
  1. The data must be processed and put into sources
  2. The visualization itself
"""

import copy
import json

import pandas as pd
from bokeh.models import ColumnDataSource, SymmetricDifferenceFilter
from bokeh.plotting import curdoc
from decipher.data import DataManager
from loguru import logger

from viz_tool.backend import BaseFilter, BooleanFilter, ExamToggleFilter, SourceManager
from viz_tool.data_ingestion import (
    CreatePersonSource,
    add_hpv_detailed_information,
    exams_pipeline,
)
from viz_tool.settings import settings

from .components import (
    get_filter_control_panel,
    get_histograms,
    get_lexis_plots,
    get_stats_div,
)


def example_app(source_manager: SourceManager):
    lexis_plots = get_lexis_plots(source_manager)

    histogram_cyt, histogram_hist, histogram_hpv, histogram_risk = get_histograms(
        source_manager
    )

    # Remove delta plot and table as these are related to predictions, which we are not doing
    # delta = DeltaScatter(source_manager)
    # table = PersonTable(source_manager)
    # table.person_table.styles = {"border": "1px solid #e6e6e6", "border-radius": "5px"}
    # table.person_table.height = 500

    # Remove vaccine filters as we do not have vaccine data
    # vaccine_type = get_filter_element_from_source_manager(
    #     "Vaccine type", source_manager
    # )
    # vaccine_group = get_filter_element_from_source_manager(
    #     "Vaccine age", source_manager
    # )
    # category_group = get_filter_element_from_source_manager("Region", source_manager)

    filter_grid = get_filter_control_panel(source_manager)

    stats_div = get_stats_div(source_manager)

    # Add names to elements for manual placement in html
    histogram_cyt.figure.name = "histogram_cyt"
    histogram_hist.figure.name = "histogram_hist"
    histogram_hpv.figure.name = "histogram_hpv"
    histogram_risk.figure.name = "histogram_risk"
    stats_div.name = "stats_div"
    for name, plot in lexis_plots.items():
        plot.figure.name = f"lexis__{name}"

    for element in (
        *(plot.figure for plot in lexis_plots.values()),
        histogram_cyt.figure,
        histogram_hist.figure,
        histogram_hpv.figure,
        histogram_risk.figure,
        stats_div,
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


def _high_risk_exam(
    exam_source_data: dict, risk_states: set[int], exam_type: str
) -> list[int]:
    return [
        i
        for i, (state, type_) in enumerate(
            zip(exam_source_data["risk"], exam_source_data["exam_type"])
        )
        if state in risk_states and type_ == exam_type
    ]


def _get_filters(source_manager: SourceManager) -> dict[str, BaseFilter]:
    hpv_exam_indices = [
        i
        for i, type in enumerate(source_manager.exam_source.data["exam_type"])
        if type == "HPV"
    ]
    hpv_16_exam_indices = [
        i
        for i, details in enumerate(
            source_manager.exam_source.data["exam_detailed_results"]
        )
        if "16" in details
    ]

    base_filters: dict[str, BaseFilter] = {
        "High risk - Histology": ExamToggleFilter(
            source_manager=source_manager,
            exam_indices=_high_risk_exam(
                source_manager.exam_source.data, HIGH_RISK_STATES, "histology"
            ),
        ),
        "High risk - Cytology": ExamToggleFilter(
            source_manager=source_manager,
            exam_indices=_high_risk_exam(
                source_manager.exam_source.data, HIGH_RISK_STATES, "cytology"
            ),
        ),
        "HPV": ExamToggleFilter(
            source_manager=source_manager,
            exam_indices=hpv_exam_indices,
        ),
        "HPV 16": ExamToggleFilter(
            source_manager=source_manager,
            exam_indices=hpv_16_exam_indices,
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


def get_exam_results(
    exams_subset: pd.DataFrame, person_df: pd.DataFrame, exam_field: str
) -> pd.Series:
    """Return a Series of exam results as lists for each person."""
    exam_results = exams_subset.groupby("PID")[exam_field].apply(lambda x: x.tolist())
    mapped_results = person_df["PID"].map(exam_results)
    mapped_results[mapped_results.isna()] = mapped_results[mapped_results.isna()].apply(
        lambda x: []
    )
    return mapped_results


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

    # Make data for histogram
    person_df["cyt_diagnosis"] = get_exam_results(
        exams_df.query("exam_type == 'cytology'"), person_df, "exam_diagnosis"
    )
    person_df["hist_diagnosis"] = get_exam_results(
        exams_df.query("exam_type == 'histology'"), person_df, "exam_diagnosis"
    )
    person_df["hpv_test_type"] = get_exam_results(
        exams_df.query("exam_type == 'HPV' & exam_diagnosis == 'positiv'"),
        person_df,
        "detailed_exam_type",
    )

    source_manager = SourceManager(
        ColumnDataSource(person_df),
        ColumnDataSource(exams_df),
    )
    source_manager.filters = _get_filters(source_manager)

    example_app(source_manager)


# Name is not main when run through bokeh serve, so no __name__ == __main__ guard
main()
