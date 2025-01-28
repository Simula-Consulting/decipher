"""Components and utilities for the pilot app."""
import itertools
from functools import partial
from typing import Collection, Hashable, cast

import numpy as np
from bokeh.layouts import column, grid, row
from bokeh.models import Div  # type: ignore  # MyPy does not find this import
from bokeh.models import InlineStyleSheet

from bokeh_demo.backend import SourceManager
from bokeh_demo.frontend import (
    HistogramPlot,
    LabelSelectedMixin,
    LexisPlot,
    get_filter_element_from_source_manager,
    get_timedelta_tick_formatter,
)
from bokeh_demo.settings import settings


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


class LexisPlotAge(LexisPlot):
    _title: str = "Age vs. Year"
    _y_label: str = "Year"
    _scatter_y_key = "exam_date"
    _lexis_line_y_key = "lexis_line_endpoints_year"
    _vaccine_line_y_key: str = "vaccine_line_endpoints_year"
    _y_axis_type = "datetime"


class HistogramWithMean(LabelSelectedMixin, HistogramPlot):
    """Histogram with mean and standard deviation label."""

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
        # We only use this class with numeric results
        # Not properly checked, so prone to user error
        selected_results = cast(
            list[int | float],
            list(
                itertools.chain.from_iterable(
                    self.results_per_person[i] for i in selected_indices
                )
            ),
        )
        mean = np.mean(selected_results)
        std = np.std(selected_results)
        return f"Mean: {mean:.2f} \n" f"Std: {std:.2f}"


### Statistics panel ###


def _get_stats_text(source_manager: SourceManager) -> str:
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
        source_manager.person_source.data["exam_time_age"][i] for i in selected_indices
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
            np.diff(ages) for ages in source_manager.person_source.data["exam_time_age"]
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


def get_stats_div(source_manager: SourceManager):
    def update_stats_text(attr, old, new):
        """Update the statistics text.

        Bokeh requires the signature of this function to be
        (attr, old, new), but we do not use any of these."""
        stats_div.text = _get_stats_text(source_manager)

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
        text=_get_stats_text(source_manager),
        styles={"color": "#555", "width": "100%"},
        stylesheets=[stylesheet],
    )
    source_manager.person_source.selected.on_change("indices", update_stats_text)  # type: ignore
    return stats_div


def get_filter_control_panel(source_manager: SourceManager):
    """Get the UI element for the filter control."""
    hpv_exam = get_filter_element_from_source_manager("HPV", source_manager)
    hpv_16_exam = get_filter_element_from_source_manager("HPV 16", source_manager)
    high_risk_hist_exam = get_filter_element_from_source_manager(
        "High risk - Histology", source_manager
    )
    high_risk_cyt_exam = get_filter_element_from_source_manager(
        "High risk - Cytology", source_manager
    )

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

    return filter_grid


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


def get_histograms(source_manager):
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

    return histogram_cyt, histogram_hist, histogram_hpv, histogram_risk


def get_lexis_plots(source_manager):
    lexis_plots = {
        "age_index": LexisPlot(source_manager),
        "age_year": LexisPlotAge(source_manager),
        "year_age": LexisPlotYearAge(source_manager),
    }

    # Link ranges
    lexis_plots["age_index"].figure.x_range = lexis_plots["age_year"].figure.x_range
    lexis_plots["year_age"].figure.y_range = lexis_plots["age_year"].figure.x_range
    lexis_plots["age_year"].figure.y_range = lexis_plots["year_age"].figure.x_range

    return lexis_plots
