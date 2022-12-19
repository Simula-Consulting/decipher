from typing import Sequence

import numpy as np

# mypy complains that bokeh.models does not have these attributes.
# We were unsuccessful in finding the origin of the bug.
# What we know is that for, for example, the DataTable, mypy correctly
# identifies it in bokeh.models.widgets.tables and also that it is exported
# to bokeh.models.widgets. However, for some reason, it is not propagated to
# bokeh.models.
from bokeh.models import (  # type: ignore
    Circle,
    CustomJSExpr,
    CustomJSHover,
    DataTable,
    HoverTool,
    Label,
    Legend,
    LegendItem,
    TableColumn,
)
from bokeh.models.tickers import FixedTicker
from bokeh.plotting import figure

from .backend import SourceManager
from .settings import settings


class ToolsMixin:
    def _get_tools(self):
        return settings.default_tools + settings.extra_tools


class LexisPlot(ToolsMixin):
    _title: str = "Lexis plot"
    _x_label: str = "Age"
    _y_label: str = "Individual #"

    _lexis_line_y_key: str = "lexis_line_endpoints_person_index"
    _lexis_line_x_key: str = "lexis_line_endpoints_age"
    _vaccine_line_x_key: str = "vaccine_line_endpoints_age"
    _vaccine_line_y_key: str = "lexis_line_endpoints_person_index"
    _scatter_y_key: str = "person_index"
    _scatter_x_key: str = "age"

    _marker_key: str = "state"
    _marker_color_key: str = "state"

    # TODO: move to config class or settings
    _markers: list[str | None] = [None, "square", "circle", "diamond"]
    _marker_colors: list[str | None] = [None, "blue", "green", "red"]
    _vaccine_line_width: int = 3
    _vaccine_line_color: str = "tan"

    def __init__(self, source_manager: SourceManager):
        self.source_manager = source_manager
        self.figure = figure(
            title=self._title,
            x_axis_label=self._x_label,
            y_axis_label=self._y_label,
            tools=self._get_tools(),
        )
        self.life_line = self.figure.multi_line(
            self._lexis_line_x_key,
            self._lexis_line_y_key,
            source=source_manager.person_source,
        )
        self.vaccine_line = self.figure.multi_line(
            self._vaccine_line_x_key,
            self._vaccine_line_y_key,
            source=source_manager.person_source,
            line_width=self._vaccine_line_width,
            color=self._vaccine_line_color,
        )

        # Legend
        # TODO: Make more general by using mixin
        # The legend layout code must come before the scatter renderer,
        # which will add new items to the legend.
        self.figure.add_layout(
            Legend(
                items=[
                    LegendItem(label="Vaccine", renderers=[self.vaccine_line], index=0)
                ],
                orientation="horizontal",
            ),
            "above",
        )
        self.scatter = self.figure.circle(
            self._scatter_x_key,
            self._scatter_y_key,
            source=self.source_manager.exam_source,
            color={
                "expr": CustomJSExpr(
                    args={"colors": self._marker_colors},
                    code=f"return this.data.{self._marker_color_key}.map(i => colors[i]);",  # noqa: E501
                )
            },
            legend_group="state_label",
        )

        # Tooltip for detailed exam data
        hover_tool = HoverTool(
            tooltips=[("Type", "@exam_type"), ("Result", "@exam_result")],
            renderers=[self.scatter],
        )
        self.figure.add_tools(hover_tool)
        # It is not possible to have the hover_glyph have a different size than normal
        # See https://github.com/bokeh/bokeh/issues/2367 (about nonselection_glyph,
        # but same issue.)
        # We get around this issue by instead having a thick line_width, which
        # gives a similar effect to having a bigger marker.
        self.scatter.hover_glyph = Circle(x="x", y="y", line_width=10, line_color="red")


class LexisPlotAge(LexisPlot):
    _y_label: str = "Year"
    _scatter_y_key = "year"
    _lexis_line_y_key = "lexis_line_endpoints_year"
    _vaccine_line_y_key: str = "vaccine_line_endpoints_year"


def get_position_list(array: Sequence) -> Sequence[int]:
    """Given an array, return the position of each element in the sorted list.

    >>> get_position_list([2, 0, 1, 4])
    [2, 0, 1, 3]
    >>> get_position_list([1, 4, 9, 2])
    [0, 2, 3, 1]
    """
    sorted_indices = (i for i, _ in sorted(enumerate(array), key=lambda iv: iv[1]))
    index_map = {n: i for i, n in enumerate(sorted_indices)}
    return [index_map[n] for n in range(len(array))]


class TrajectoriesPlot(ToolsMixin):
    _exam_color: str = "blue"
    _predicted_exam_color: str = "red"

    def __init__(self, source_manager: SourceManager):
        self.figure = figure(x_axis_label="Age", tools=self._get_tools())

        self.exam_plot = self.figure.multi_line(
            "exam_time_age",
            "exam_results",
            source=source_manager.person_source,
            view=source_manager.only_selected_view,
            color=self._exam_color,
            legend_label="Actual observation",
        )
        self.predicted_exam_plot = self.figure.multi_line(
            "exam_time_age",
            "predicted_exam_results",
            source=source_manager.person_source,
            view=source_manager.only_selected_view,
            color=self._predicted_exam_color,
            legend_label="Predicted observation",
        )

        # Simple tooltip
        list_formatter = CustomJSHover(
            code="""
        return `[${value.map(n => n.toFixed(2)).join(', ')}]`
        """
        )
        hover_tool = HoverTool(
            tooltips=[
                ("Id", "$index"),
                ("Vaccine", "@vaccine_age{0.0}"),
                ("Probabilities", "@prediction_probabilities{custom}"),
            ],
            formatters={"@prediction_probabilities": list_formatter},
        )
        self.figure.add_tools(hover_tool)

        # Set y-ticks to state names
        self.figure.yaxis.ticker = FixedTicker(
            ticks=list(range(len(settings.label_map)))
        )
        self.figure.yaxis.major_label_overrides = dict(enumerate(settings.label_map))


class DeltaScatter(ToolsMixin):
    _delta_scatter_x_key: str = "deltascatter__delta_score_index"
    _delta_scatter_y_key: str = "delta"

    def __init__(self, source_manager: SourceManager):
        self.figure = figure(
            x_axis_label="Individual",
            y_axis_label="Delta score (lower better)",
            tools=self._get_tools(),
        )

        # Generate a index list based on delta score
        # TODO: consider guard for overwrite
        source_manager.person_source.data[
            "deltascatter__delta_score_index"
        ] = get_position_list(source_manager.person_source.data["delta"])

        self.scatter = self.figure.scatter(
            self._delta_scatter_x_key,
            self._delta_scatter_y_key,
            source=source_manager.person_source,
        )


class PersonTable:
    def __init__(self, source_manager: SourceManager):
        # Add column for correct state and prediction discrepancy
        exam_results = source_manager.person_source.data["exam_results"]
        exam_times = source_manager.person_source.data["prediction_time"]
        true_state_at_prediction = [
            exam_result[exam_time]
            for exam_result, exam_time in zip(exam_results, exam_times)
        ]
        prediction_discrepancy = [
            true - predicted
            for true, predicted in zip(
                true_state_at_prediction,
                source_manager.person_source.data["predicted_exam_result"],
            )
        ]

        source_manager.person_source.data[
            "persontable__true_state"
        ] = true_state_at_prediction
        source_manager.person_source.data[
            "persontable__discrepancy"
        ] = prediction_discrepancy

        self.person_table = DataTable(
            source=source_manager.person_source,
            columns=[
                TableColumn(title="Delta score", field="delta"),
                TableColumn(title="Predicted state", field="predicted_exam_result"),
                TableColumn(title="Correct state", field="persontable__true_state"),
                TableColumn(
                    title="Prediction discrepancy", field="persontable__discrepancy"
                ),
            ],
            styles={
                "border": "1px solid black",
                "margin-right": "40px",
            },
        )


class LabelSelectedMixin:
    def add_label(self):
        self.label = Label(
            x=10,
            y=410,
            x_units="screen",
            y_units="screen",
            text=self._get_label_text(),
            text_font_size="12px",
            border_line_color="black",
            border_line_alpha=1.0,
            background_fill_color="white",
            background_fill_alpha=1.0,
        )
        self.figure.add_layout(self.label)

    def _get_age_at_exam(self, selected_indices):
        return (
            [
                age
                for age, state in zip(
                    self.source_manager.person_source.data["exam_time_age"][i],
                    self.source_manager.person_source.data["exam_results"][i],
                )
                if state != 0
            ]
            for i in selected_indices
        )

    @staticmethod
    def _compute_average_screening_interval(nested_age_at_exam):
        screening_intervals = []
        for x in nested_age_at_exam:
            screening_intervals += np.diff(x).tolist()
        # Convert to months
        return np.mean(screening_intervals) * 12

    def _get_label_text(self, selected_indices=None):
        selected_indices = selected_indices or range(self._number_of_individuals)
        n_vaccines = sum(
            self.source_manager.person_source.data["vaccine_age"][i] is not None
            for i in selected_indices
        )

        nested_age_at_exam = self._get_age_at_exam(selected_indices)
        average_screening_interval = self._compute_average_screening_interval(
            nested_age_at_exam
        )
        return (
            f" Individuals selected: {len(selected_indices)} \n"
            f" Individuals with vaccinations: {n_vaccines} \n"
            f" Average screening interval: ~{round(average_screening_interval, 2)} months"  # noqa: E501
        )

    def get_update_label_callback(self):
        def update_label_callback(attr, old, new):
            new = new if len(new) else list(range(self._number_of_individuals))
            self.label.text = self._get_label_text(new)

        return update_label_callback


class HistogramPlot(ToolsMixin, LabelSelectedMixin):
    def __init__(self, source_manager: SourceManager):
        self.source_manager = source_manager
        self._number_of_individuals = len(
            self.source_manager.person_source.data["index"]
        )

        self.figure = figure(tools=self._get_tools())
        self.quad = self.figure.quad(
            top=self.compute_histogram_data(),
            bottom=0,
            left=np.arange(0, 4) + 0.5,
            right=np.arange(1, 5) + 0.5,
            fill_color="navy",
            line_color="white",
            alpha=0.5,
            name="quad",
        )

        self.source_manager.person_source.selected.on_change(
            "indices", self.get_update_histogram_callback()
        )

        # Add label from LabelSelectedMixin
        self.add_label()
        self.source_manager.person_source.selected.on_change(
            "indices", self.get_update_label_callback()
        )

        self._set_properties()

    def _set_properties(self):
        properties = {
            "y_range": {"start": 0},
            "xaxis": {
                "axis_label": "State",
                "ticker": list(range(len(settings.label_map))),
                "major_label_overrides": dict(enumerate(settings.label_map)),
            },
            "yaxis": {"axis_label": "Count"},
            "grid": {"grid_line_color": "white"},
        }

        for module, module_options in properties.items():
            for option, value in module_options.items():
                setattr(getattr(self.figure, module), option, value)

    def compute_histogram_data(self, selected_indices=None):
        selected_indices = selected_indices or range(self._number_of_individuals)
        state_occurrences = self._count_state_occurrences(
            [
                [
                    yi
                    for i in selected_indices
                    for yi in self.source_manager.person_source.data["exam_results"][i]
                    if yi != 0
                ]
            ]
        )
        return [
            value for _, value in sorted(state_occurrences.items(), key=lambda x: x[0])
        ]

    @staticmethod
    def _count_state_occurrences(nested_list_of_states):
        out = {1: 0, 2: 0, 3: 0, 4: 0}
        for list_of_states in nested_list_of_states:
            for state in list_of_states:
                out[state] += 1
        return out

    def get_update_histogram_callback(self):
        def update_histogram(attr, old, new):
            new = new if len(new) else list(range(self._number_of_individuals))

            self.quad.data_source.data["top"] = self.compute_histogram_data(new)

        return update_histogram