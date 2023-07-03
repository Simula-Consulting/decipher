"""Components and utilities for the pilot app."""
import itertools
from typing import Collection, Hashable, cast

import numpy as np

from bokeh_demo.backend import SourceManager
from bokeh_demo.frontend import (
    HistogramPlot,
    LabelSelectedMixin,
    LexisPlot,
    get_timedelta_tick_formatter,
)


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
