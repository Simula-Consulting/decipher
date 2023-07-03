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
from typing import Hashable

import numpy as np
import pandas as pd
from bokeh.layouts import column, grid, row
from bokeh.models import Div  # type: ignore  # Missing type hint
from bokeh.models import ColumnDataSource, InlineStyleSheet, SymmetricDifferenceFilter
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
    LexisPlot,
    get_filter_element_from_source_manager,
)
from bokeh_demo.settings import settings

from .components import HistogramWithMean, LexisPlotAge, LexisPlotYearAge

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


def _link_ranges(lexis_plots):
    lexis_plots["age_index"].figure.x_range = lexis_plots["age_year"].figure.x_range
    lexis_plots["year_age"].figure.y_range = lexis_plots["age_year"].figure.x_range
    lexis_plots["age_year"].figure.y_range = lexis_plots["year_age"].figure.x_range


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

    def _risk_label(risk_level: Hashable) -> str:
        """Format risk as 'label (risk_level)', e.g. 'Normal (1)'"""
        if not isinstance(risk_level, int):
            return str(risk_level)
        return f"{settings.label_map[risk_level]} ({risk_level})"

    histogram_risk = HistogramWithMean.from_person_field(
        source_manager, "exam_results", label_mapper=_risk_label
    )

    # Set up labels and titles
    histogram_cyt.figure.title.text = "Cytology diagnosis"  # type: ignore
    histogram_hist.figure.title.text = "Histology diagnosis"  # type: ignore
    histogram_hpv.figure.title.text = "HPV test type"  # type: ignore
    histogram_risk.figure.title.text = "Exam risk levels"  # type: ignore
    histogram_cyt.figure.xaxis.axis_label = None
    histogram_hist.figure.xaxis.axis_label = None
    histogram_hpv.figure.xaxis.axis_label = None
    histogram_hist.figure.xaxis.axis_label = None

    # Adjust label positions
    histogram_risk.label.y -= 30  # type: ignore

    # Remove delta plot and table as these are related to predictions, which we are not doing
    # delta = DeltaScatter(source_manager)
    # table = PersonTable(source_manager)
    # table.person_table.styles = {"border": "1px solid #e6e6e6", "border-radius": "5px"}
    # table.person_table.height = 500

    hpv_exam = get_filter_element_from_source_manager("HPV", source_manager)
    hpv_16_exam = get_filter_element_from_source_manager("HPV 16", source_manager)
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
            row([Div(), Div(text="Active"), Div(text="Invert"), Div(text="On person")]),
            hpv_exam,
            hpv_16_exam,
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
    ]  # type: ignore

    ## Statistics ##
    def _get_stats_text() -> str:
        """Get text for statistics div."""
        number_of_individuals = len(source_manager.person_source.data["PID"])
        # If nothing is selected, interpret it as everything is selected.
        selected_indices = source_manager.person_source.selected.indices or range(  # type: ignore
            number_of_individuals
        )
        number_of_selected = len(selected_indices)

        # Exam info
        number_of_exams = len(source_manager.exam_source.selected.indices) or len(  # type: ignore
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

        person_icon = """
        <svg class="icon" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="currentColor" viewBox="0 0 20 19">
        <path d="M14.5 0A3.987 3.987 0 0 0 11 2.1a4.977 4.977 0 0 1 3.9 5.858A3.989 3.989 0 0 0 14.5 0ZM9 13h2a4 4 0 0 1 4 4v2H5v-2a4 4 0 0 1 4-4Z"/>
        <path d="M5 19h10v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2ZM5 7a5.008 5.008 0 0 1 4-4.9 3.988 3.988 0 1 0-3.9 5.859A4.974 4.974 0 0 1 5 7Zm5 3a3 3 0 1 0 0-6 3 3 0 0 0 0 6Zm5-1h-.424a5.016 5.016 0 0 1-1.942 2.232A6.007 6.007 0 0 1 17 17h2a1 1 0 0 0 1-1v-2a5.006 5.006 0 0 0-5-5ZM5.424 9H5a5.006 5.006 0 0 0-5 5v2a1 1 0 0 0 1 1h2a6.007 6.007 0 0 1 4.366-5.768A5.016 5.016 0 0 1 5.424 9Z"/>
        </svg>"""

        clipboard_icon = """
        <svg class="icon" xmlns="http://www.w3.org/2000/svg" fill="currentColor" viewBox="0 0 18 20">
        <path d="M16 1h-3.278A1.992 1.992 0 0 0 11 0H7a1.993 1.993 0 0 0-1.722 1H2a2 2 0 0 0-2 2v15a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V3a2 2 0 0 0-2-2ZM7 2h4v3H7V2Zm5.7 8.289-3.975 3.857a1 1 0 0 1-1.393 0L5.3 12.182a1.002 1.002 0 1 1 1.4-1.436l1.328 1.289 3.28-3.181a1 1 0 1 1 1.392 1.435Z"/>
        </svg>"""

        left_right_icon = """
        <svg class="icon" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 16 14">
        <path stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 10H1m0 0 3-3m-3 3 3 3m1-9h10m0 0-3 3m3-3-3-3"/>
        </svg>"""

        age_icon = """
        <svg class="icon" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-6 h-6">
        <path stroke-linecap="round" stroke-linejoin="round" d="M15 9h3.75M15 12h3.75M15 15h3.75M4.5 19.5h15a2.25 2.25 0 002.25-2.25V6.75A2.25 2.25 0 0019.5 4.5h-15a2.25 2.25 0 00-2.25 2.25v10.5A2.25 2.25 0 004.5 19.5zm6-10.125a1.875 1.875 0 11-3.75 0 1.875 1.875 0 013.75 0zm1.294 6.336a6.721 6.721 0 01-3.17.789 6.721 6.721 0 01-3.168-.789 3.376 3.376 0 016.338 0z" />
        </svg>"""

        def _format_stats(name, value, help=None):
            help = (
                f"<span class='help'>?<span class='tooltip'>{help}</span></span>"
                if help
                else ""
            )
            return f"<div class='stats'><span class='name'>{name}{help}</span> <span>{value}</span></div>"

        statistics_list = "\n".join(
            _format_stats(name, value, help)
            for name, value, help in (
                (
                    person_icon,
                    f"{number_of_selected} / {number_of_individuals}",
                    "Number of individuals",
                ),
                (clipboard_icon, number_of_exams, "Number of exams"),
                (
                    left_right_icon,
                    f"{screening_interval_mean:.2f} ± {screening_interval_std:.2f} years",
                    "Screening interval",
                ),
                (
                    f"{clipboard_icon} / {person_icon}",
                    f"{number_of_exams_mean:.2f} ± {number_of_exams_std:.2f}",
                    "Number of exams per individual",
                ),
                (age_icon, f"{ages_mean:.2f} ± {ages_std:.2f} years", "Age at exam"),
            )
        )
        return (
            "<h2>Selection statistics</h2>"
            "<div class='stats-group' style=''>"
            f"{statistics_list}"
            "</div>"
        )

    def update_stats_text(attr, old, new):
        """Update the statistics text.

        Bokeh requires the signature of this function to be
        (attr, old, new), but we do not use any of these."""
        stats_div.text = _get_stats_text()

    stylesheet = InlineStyleSheet(
        css="""
        .stats-group {
            display: grid;
            grid-column-gap: 20px;
            grid-template-columns: 1fr 1fr 1fr;
            font-size: 18px;
            width: 80%;
            margin: 0 auto;
        }
        .stats { border-bottom: 1px solid #bdbdbd; display: flex; line-height: 2em; margin-top: 5px;}
        .stats span.name { flex-grow: 1; }
        .tooltip { display: none; position: absolute; bottom: 2em; background: #686868; color: white; padding: 5px 10px; border-radius: 4px; opacity: 0; transition: opacity 0.3s; width: max-content;}
        .help { position: relative; padding: 0 4px; border-radius: 100px; border: 1px solid black; width: 1.2em; height: 1.2em; line-height: 1.2em; display: inline-block; text-align: center; margin-left: 8px; vertical-align: super; font-size: smaller;}
        .help:hover .tooltip { opacity: 1; display: block;}
        .icon { width: 1.2em; height: 1.2em; vertical-align: middle; }
        .bk-clearfix { width: 100% }
        """
    )
    stats_div = Div(
        text=_get_stats_text(),
        styles={"color": "#555", "width": "100%"},
        stylesheets=[stylesheet],
    )
    source_manager.person_source.selected.on_change("indices", update_stats_text)  # type: ignore

    # Add names to elements for manual placement in html
    histogram_cyt.figure.name = "histogram_cyt"
    histogram_hist.figure.name = "histogram_hist"
    histogram_hpv.figure.name = "histogram_hpv"
    histogram_risk.figure.name = "histogram_risk"
    stats_div.name = "stats_div"

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
