# type: ignore
"""Example Bokeh server app.

The app consist of two main "parts"
  1. The data must be processed and put into sources
  2. The visualization itself
"""

import copy
import itertools
import json
from enum import Enum
from functools import partial
from typing import Collection, Hashable

import numpy as np
import pandas as pd
from bokeh.layouts import column, grid, row
from bokeh.models import ColumnDataSource, Div, SymmetricDifferenceFilter
from bokeh.plotting import curdoc
from decipher.data import DataManager
from loguru import logger

from bokeh_demo.backend import (
    BaseFilter,
    BooleanFilter,
    ExamToggleFilter,
    SourceManager,
)
from bokeh_demo.data_ingestion import (
    CreatePersonSource,
    add_hpv_detailed_information,
    exams_pipeline,
)
from bokeh_demo.frontend import (
    HistogramPlot,
    LabelSelectedMixin,
    LexisPlot,
    LexisPlotAge,
    get_filter_element_from_source_manager,
    get_timedelta_tick_formatter,
)
from bokeh_demo.settings import settings

DIAGNOSIS_ABBREVIATIONS = {
    "Normal m betennelse eller blod": "Normal w.b",
    "Cancer Cervix cancer andre/usp": "Cancer",
    "Normal uten sylinder": "Normal u s",
}
"""Abbreviations for diagnosis names."""

HPV_TEST_ABBREVIATIONS = {
    "Abbott RealTime High Risk HPV": "Abbott HR",
    "Cobas 4800 System": "Cobas 4800",
}


def try_abbreviate(abbreviations: dict[str, str], diagnosis: str) -> str:
    """Abbreviates diagnosis names if they are in the DIAGNOSIS_ABBREVIATIONS dictionary."""
    return abbreviations.get(diagnosis, diagnosis)


class LexisPlotYearAge(LexisPlot):
    """Lexis plot with year on x-axis and age on y-axis."""

    _title: str = "Year vs. Age"

    _y_label: str = "Age"
    _scatter_y_key: str = "age"
    _x_label: str = "Year"
    _scatter_x_key: str = "exam_date"

    _lexis_line_x_key: str = "lexis_line_endpoints_year"
    _lexis_line_y_key: str = "lexis_line_endpoints_age"
    _vaccine_line_x_key: str = "vaccine_line_endpoints_year"
    _vaccine_line_y_key: str = "vaccine_line_endpoints_age"

    _y_axis_type = "linear"
    _x_axis_type = "datetime"

    _y_axis_tick_format_getter = get_timedelta_tick_formatter
    _x_axis_tick_format_getter = None


def _link_ranges(lexis_plots):
    lexis_plots["age_index"].figure.x_range = lexis_plots["age_year"].figure.x_range
    lexis_plots["year_age"].figure.y_range = lexis_plots["age_year"].figure.x_range
    lexis_plots["age_year"].figure.y_range = lexis_plots["year_age"].figure.x_range


class HistogramWithMean(LabelSelectedMixin, HistogramPlot):
    def __init__(
        self,
        results_per_person: list[list[Hashable]],
        class_list: list[Hashable] | dict[Hashable, str],
        source_manager: SourceManager,
    ):
        super().__init__(results_per_person, class_list, source_manager)
        # Add label from LabelSelectedMixin
        self.register_label()

    def _get_label_text(self, selected_indices: Collection[int]) -> str:
        selected_results = list(
            itertools.chain.from_iterable(
                self.results_per_person[i] for i in selected_indices
            )
        )
        mean = np.mean(selected_results)
        std = np.std(selected_results)
        return f"Mean: {mean:.2f} \n" f"Std: {std:.2f}"


def example_app(source_manager: SourceManager):
    lexis_plots = {
        "age_index": LexisPlot(source_manager),
        "age_year": LexisPlotAge(source_manager),
        "year_age": LexisPlotYearAge(source_manager),
    }
    _link_ranges(lexis_plots)
    # Add names for reference in the HTML template
    for name, plot in lexis_plots.items():
        plot.figure.name = f"lexis__{name}"

    histogram_cyt = HistogramPlot.from_person_field(
        source_manager,
        "cyt_diagnosis",
        label_mapper=partial(try_abbreviate, DIAGNOSIS_ABBREVIATIONS),
    )
    histogram_hist = HistogramPlot.from_person_field(source_manager, "hist_diagnosis")
    histogram_hpv = HistogramPlot.from_person_field(
        source_manager,
        "hpv_test_type",
        label_mapper=partial(try_abbreviate, HPV_TEST_ABBREVIATIONS),
    )

    def _risk_label(risk_level: int) -> str:
        """Format risk as 'label (risk_level)', e.g. 'Normal (1)'"""
        return f"{settings.label_map[risk_level]} ({risk_level})"

    histogram_risk = HistogramWithMean.from_person_field(
        source_manager, "exam_results", label_mapper=_risk_label
    )

    # Set up labels and titles
    histogram_cyt.figure.title.text = "Cytology diagnosis"
    histogram_hist.figure.title.text = "Histology diagnosis"
    histogram_hpv.figure.title.text = "HPV test type"
    histogram_risk.figure.title.text = "Exam risk levels"
    histogram_cyt.figure.xaxis.axis_label = None
    histogram_hist.figure.xaxis.axis_label = None
    histogram_hpv.figure.xaxis.axis_label = None
    histogram_hist.figure.xaxis.axis_label = None

    # Adjust label positions
    histogram_risk.label.y -= 20

    # Remove delta plot and table as these are related to predictions, which we are not doing
    # delta = DeltaScatter(source_manager)
    # table = PersonTable(source_manager)
    # table.person_table.styles = {"border": "1px solid #e6e6e6", "border-radius": "5px"}
    # table.person_table.height = 500

    hpv_exam = get_filter_element_from_source_manager("HPV", source_manager)
    high_risk_hist_exam = get_filter_element_from_source_manager(
        "High risk - Histology", source_manager
    )
    high_risk_cyt_exam = get_filter_element_from_source_manager(
        "High risk - Cytology", source_manager
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
            row(Div(), Div(text="Active"), Div(text="Invert"), Div(text="On person")),
            hpv_exam,
            high_risk_hist_exam,
            high_risk_cyt_exam,
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

    ## Statistics ##
    def _get_stats_text() -> str:
        """Get text for statistics div."""
        number_of_individuals = len(source_manager.person_source.data["PID"])
        # If nothing is selected, interpret it as everything is selected.
        selected_indices = source_manager.person_source.selected.indices or range(
            number_of_individuals
        )
        number_of_selected = len(selected_indices)

        # Exam info
        number_of_exams = len(source_manager.exam_source.selected.indices) or len(
            source_manager.exam_source.data["person_index"]
        )  # "person_index" chosen arbitrarily, can be any key
        ages_per_person = [
            source_manager.person_source.data["exam_time_age"][i]
            for i in selected_indices
        ]
        number_of_exams_per_person = [len(ages) for ages in ages_per_person]
        number_of_exams_mean = np.mean(number_of_exams_per_person)
        number_of_exams_std = np.std(number_of_exams_per_person)

        ages_list = list(itertools.chain.from_iterable(ages_per_person))
        ages_mean = np.mean(ages_list)
        ages_std = np.std(ages_list)

        # Screening interval
        screening_intervals = list(
            itertools.chain.from_iterable(
                np.diff(ages)
                for ages in source_manager.person_source.data["exam_time_age"]
            )
        )
        screening_interval_mean = np.mean(screening_intervals)
        screening_interval_std = np.std(screening_intervals)

        return (
            "<h2>Statistics</h2>"
            "<ul>"
            f"<li>Selected: {number_of_selected} / {number_of_individuals}</li>"
            f"<li>Exams: {number_of_exams}</li>"
            f"<li>Screening interval: {screening_interval_mean:.2f} ± {screening_interval_std:.2f} years</li>"
            f"<li>Number of exams per person: {number_of_exams_mean:.2f} ± {number_of_exams_std:.2f}</li>"
            f"<li>Age at exam: {ages_mean:.2f} ± {ages_std:.2f} years</li>"
            "</ul>"
        )

    def update_stats_text(attr, old, new):
        """Update the statistics text.

        Bokeh requires the signature of this function to be
        (attr, old, new), but we do not use any of these."""
        stats_div.text = _get_stats_text()

    stats_div = Div(text=_get_stats_text())
    source_manager.person_source.selected.on_change("indices", update_stats_text)

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

    base_filters = {
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


def _get_exam_diagnosis(exams_subset):
    return exams_subset.groupby("PID")["exam_diagnosis"].apply(lambda x: x.values)


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
    person_df["cyt_diagnosis"] = _get_exam_diagnosis(
        exams_df.query("exam_type == 'cytology'")
    ).reindex(person_df.index, fill_value=[])
    person_df["hist_diagnosis"] = _get_exam_diagnosis(
        exams_df.query("exam_type == 'histology'")
    ).reindex(person_df.index, fill_value=[])
    person_df["hpv_test_type"] = (
        exams_df.query("exam_type == 'HPV' & exam_diagnosis == 'positiv'")
        .groupby("PID")["detailed_exam_type"]
        .apply(lambda x: x.values)
        .reindex(person_df.index, fill_value=[])
    )

    source_manager = SourceManager(
        ColumnDataSource(person_df),
        ColumnDataSource(exams_df),
    )
    source_manager.filters = _get_filters(source_manager)

    example_app(source_manager)


# Name is not main when run through bokeh serve, so no __name__ == __main__ guard
main()
