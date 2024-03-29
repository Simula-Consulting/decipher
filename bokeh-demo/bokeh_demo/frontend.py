import itertools
from collections import Counter
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Collection,
    Generator,
    Hashable,
    Iterable,
    Sequence,
    cast,
)

import numpy as np

# mypy complains that bokeh.models does not have these attributes.
# We were unsuccessful in finding the origin of the bug.
# What we know is that for, for example, the DataTable, mypy correctly
# identifies it in bokeh.models.widgets.tables and also that it is exported
# to bokeh.models.widgets. However, for some reason, it is not propagated to
# bokeh.models.
from bokeh.layouts import column, grid, row
from bokeh.models import (  # type: ignore
    Circle,
    CustomJSExpr,
    CustomJSTickFormatter,
    DataTable,
    HoverTool,
    Label,
    LayoutDOM,
    Legend,
    LegendItem,
    MultiChoice,
    Paragraph,
    Quad,
    RangeSlider,
    Row,
    Switch,
    TableColumn,
)
from bokeh.models.tickers import FixedTicker
from bokeh.plotting import figure

from .backend import (
    BaseFilter,
    BooleanFilter,
    CategoricalFilter,
    ExamSimpleFilter,
    PersonSimpleFilter,
    RangeFilter,
    SimpleFilter,
    SourceManager,
    parse_filter_to_indices,
)
from .settings import settings


def pad_range(
    range: tuple[float, float], padding: float | None = None
) -> tuple[float, float]:
    """Util tool for padding a range.

    Given a range, pad such that the new interval is 1 + padding bigger than the original.

    Example:
        >>> pad_range((0, 1), 0.5)
        (-0.25, 1.25)
    """
    padding = padding if padding is not None else settings.range_padding
    min, max = range
    diff = max - min
    return (min - diff * padding / 2, max + diff * padding / 2)


class ToolsMixin:
    def _get_tools(self):
        return settings.default_tools + settings.extra_tools


def get_timedelta_tick_formatter() -> CustomJSTickFormatter:
    """Format a timedelta axis to years.

    Must be a factory, if not multiple documents at the same time will fail."""
    return CustomJSTickFormatter(
        code="""
            const millisecondsInYear = 365.25 * 24 * 60 * 60 * 1000;
            return (tick / millisecondsInYear).toFixed(1);
        """
    )


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
    _y_axis_type: str = "linear"
    _x_axis_type: str = "datetime"

    _y_axis_tick_format_getter = None
    _x_axis_tick_format_getter = get_timedelta_tick_formatter

    _marker_key: str = "risk"
    _marker_color_key: str = "risk"

    # TODO: move to config class or settings
    _markers: list[str | None] = [None, "square", "circle", "diamond"]
    _marker_colors: dict[int | float, str] = settings.color_palette
    _vaccine_line_width: int = 3
    _vaccine_line_color: str = "rgba(143, 148, 9, 0.5)"

    def __init__(self, source_manager: SourceManager):
        self.source_manager = source_manager
        self.figure = figure(
            title=self._title,
            x_axis_label=self._x_label,
            y_axis_label=self._y_label,
            y_axis_type=self._y_axis_type,
            x_axis_type=self._x_axis_type,
            tools=self._get_tools(),
            x_range=pad_range(
                self.get_min_max((self._lexis_line_x_key, self._vaccine_line_x_key))
            ),
            y_range=pad_range(
                self.get_min_max((self._lexis_line_y_key, self._vaccine_line_y_key))
            ),
        )
        if self.__class__._x_axis_tick_format_getter is not None:
            self.figure.xaxis.formatter = self.__class__._x_axis_tick_format_getter()
        if self.__class__._y_axis_tick_format_getter is not None:
            self.figure.yaxis.formatter = self.__class__._y_axis_tick_format_getter()
        self.life_line = self.figure.multi_line(
            self._lexis_line_x_key,
            self._lexis_line_y_key,
            source=source_manager.person_source,
            view=source_manager.view,
        )
        # self.vaccine_line = self.figure.multi_line(
        #     self._vaccine_line_x_key,
        #     self._vaccine_line_y_key,
        #     source=source_manager.person_source,
        #     view=source_manager.view,
        #     line_width=self._vaccine_line_width,
        #     color=self._vaccine_line_color,
        # )

        # Legend
        # TODO: Make more general by using mixin
        # Adding legend entries automatically through `legend_group` causes issues
        # with both the legend and canvas.
        # See
        #  - https://github.com/bokeh/bokeh/issues/12718 Canvas issue
        #  - https://github.com/bokeh/bokeh/issues/8010 Legend color issue
        # We here implement a workaround, by manually making the legend.
        # In order to add a new legend item, there must be an associated renderer.
        # We here create some just for the sake of the legend, and set them to be invisible.
        self.figure.add_layout(
            Legend(
                items=[
                    # LegendItem(label="Vaccine", renderers=[self.vaccine_line], index=0),
                    *(
                        LegendItem(
                            label=settings.label_map[risk_level],
                            renderers=[
                                self.figure.circle(
                                    [0],
                                    [0],
                                    color=[self._marker_colors[risk_level]],
                                    visible=False,
                                )
                            ],
                        )
                        for risk_level in range(1, 5)
                    ),
                ],
                orientation="horizontal",
            ),
            "above",
        )
        self.scatter = self.figure.circle(
            self._scatter_x_key,
            self._scatter_y_key,
            source=self.source_manager.exam_source,
            view=source_manager.exam_view,
            color={
                "expr": CustomJSExpr(
                    args={"colors": self._marker_colors},
                    code=f"return Array.from(this.data.{self._marker_color_key}).map(i => colors.get(i));",  # noqa: E501
                )
            },
        )

        # Tooltip for detailed exam data
        hover_tool = HoverTool(
            tooltips=[
                ("Type", "@detailed_exam_type"),
                ("Result", "@exam_detailed_results"),
            ],
            renderers=[self.scatter],
        )
        self.figure.add_tools(hover_tool)
        # It is not possible to have the hover_glyph have a different size than normal
        # See https://github.com/bokeh/bokeh/issues/2367 (about nonselection_glyph,
        # but same issue.)
        # We get around this issue by instead having a thick line_width, which
        # gives a similar effect to having a bigger marker.
        self.scatter.hover_glyph = Circle(x="x", y="y", line_width=10, line_color="red")

    def get_min_max(self, keys: Iterable[str]) -> tuple[int, int] | tuple[float, float]:
        def flatten(itr: Iterable) -> Iterable:
            for element in itr:
                if isinstance(element, Iterable):
                    yield from flatten(element)
                else:
                    yield element

        x_ranges = list(
            flatten(self.source_manager.person_source.data[key] for key in keys)
        )
        return (min(x_ranges), max(x_ranges))


class LexisPlotAge(LexisPlot):
    _y_label: str = "Year"
    _scatter_y_key = "exam_date"
    _lexis_line_y_key = "lexis_line_endpoints_year"
    _vaccine_line_y_key: str = "vaccine_line_endpoints_year"
    _y_axis_type = "datetime"


def get_position_list(array: Sequence) -> Sequence[int]:
    """Given an array, return the position of each element in the sorted list.

    >>> get_position_list([2, 0, 1, 4])
    [2, 0, 1, 3]
    >>> get_position_list([1, 4, 9, 2])
    [0, 2, 3, 1]
    """
    return list(np.argsort(np.argsort(array)))


class TrajectoriesPlot(ToolsMixin):
    _exam_color: str = settings.color_palette[1]
    _predicted_exam_color: str = settings.color_palette[3]

    def __init__(self, source_manager: SourceManager):
        # Find min/max on x-axis
        x_axis_data = list(
            itertools.chain.from_iterable(
                source_manager.person_source.data["exam_time_age"]
            )
        )

        self.figure = figure(
            x_axis_label="Age",
            tools=self._get_tools(),
            x_range=pad_range((min(x_axis_data), max(x_axis_data))),
            y_range=pad_range((1, len(settings.label_map) - 1)),
        )

        self.exam_plot = self.figure.multi_line(
            "exam_time_age",
            "exam_results",
            source=source_manager.person_source,
            view=source_manager.combined_view,
            color=self._exam_color,
            legend_label="Observation",
        )
        # self.predicted_exam_plot = self.figure.multi_line(
        #     "exam_time_age",
        #     "predicted_exam_results",
        #     source=source_manager.person_source,
        #     view=source_manager.combined_view,
        #     color=self._predicted_exam_color,
        #     legend_label="Predicted observation",
        # )

        # Simple tooltip
        # list_formatter = CustomJSHover(
        #     code="""
        # return `[${value.map(n => n.toFixed(2)).join(', ')}]`
        # """
        # )
        hover_tool = HoverTool(
            tooltips=[
                ("Id", "$index"),
                ("Vaccine", "@vaccine_age{0.0} (@vaccine_type)"),
                # ("Probabilities", "@prediction_probabilities{custom}"),
            ],
            # formatters={"@prediction_probabilities": list_formatter},
        )
        self.figure.add_tools(hover_tool)

        # Set y-ticks to state names
        self.figure.yaxis.ticker = FixedTicker(
            ticks=list(range(len(settings.label_map)))
        )
        self.figure.yaxis.major_label_overrides = settings.label_map


class DeltaScatter(ToolsMixin):
    _delta_scatter_x_key: str = "deltascatter__delta_score_index"
    _delta_scatter_y_key: str = "delta"

    def __init__(self, source_manager: SourceManager):
        number_of_individuals = len(source_manager.person_source.data["index"])
        self.figure = figure(
            x_axis_label="Individual",
            y_axis_label="Delta score (lower better)",
            tools=self._get_tools(),
            y_range=pad_range((-1, 1)),
            x_range=pad_range((0, number_of_individuals)),
        )

        # Generate a index list based on delta score
        # TODO: consider guard for overwrite
        source_manager.person_source.data[
            "deltascatter__delta_score_index"
        ] = get_position_list(
            cast(Sequence[int], source_manager.person_source.data["delta"])
        )

        self.scatter = self.figure.scatter(
            self._delta_scatter_x_key,
            self._delta_scatter_y_key,
            source=source_manager.person_source,
            view=source_manager.view,
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
        )


class LabelSelectedMixin:
    source_manager: SourceManager
    _number_of_individuals: int

    def add_label(self):
        self.label = Label(
            x=10,
            y=470,
            x_units="screen",
            y_units="screen",
            text=self._get_label_text(range(self._number_of_individuals)),
            text_font_size="12px",
            border_line_color="black",
            border_line_alpha=1.0,
            background_fill_color="white",
            background_fill_alpha=1.0,
        )
        self.figure.add_layout(self.label)

    def register_label(self) -> None:
        """Add label to layout and attach callbacks."""
        self.add_label()
        self.source_manager.person_source.selected.on_change(  # type: ignore
            "indices", self.get_update_label_callback()
        )
        self.source_manager.view.on_change("filter", self.get_update_label_callback())

    def _get_age_at_exam(
        self, selected_indices: Iterable[int]
    ) -> Generator[list[float], None, None]:
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
    def _compute_average_screening_interval(
        nested_age_at_exam: Iterable[Sequence[float]],
    ) -> float:
        screening_intervals: list[float] = []
        for x in nested_age_at_exam:
            screening_intervals += np.diff(x).tolist()
        # Convert to months
        return cast(float, np.mean(screening_intervals)) * 12

    def _get_label_text(self, selected_indices: Collection[int]) -> str:
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

    def get_update_label_callback(self) -> Callable[..., None]:
        """Get a callback function for updating the label.

        The label update callback is attached to multiple quantities, so we do not
        use the supplied old/new values, but use directly the state in source_manager.
        """

        def update_label_callback(attr, old, new) -> None:
            # If nothing is selected, interpret it as everything is selected.
            selected_indices = set(
                self.source_manager.person_source.selected.indices  # type: ignore
                or range(self._number_of_individuals)
            )
            filtered_indices = set(
                parse_filter_to_indices(
                    self.source_manager.view.filter, self._number_of_individuals  # type: ignore
                )
            )
            filtered_and_selected_indices = selected_indices & filtered_indices
            self.label.text = self._get_label_text(filtered_and_selected_indices)

        return update_label_callback


class HistogramPlot(LabelSelectedMixin):
    """Display a histogram of the selected results.

    The selection and filters will be applied to the outer list, yielding only
    results for the selected and filtered individuals.

    Args:
        results_per_person: the outer list corresponds to the individuals, and the inner
          list corresponds to the results to be plotted. The selection and filters will
          be applied to the outer list, yielding only results for the selected and
          filtered individuals.
        class_list: List of classes to be plotted. Does not need to include all the
          classes in the data. Histogram plotted in the order the classes are given.
          If a dictionary is given, the keys are the classes and the values are the
          labels on the ticks.
        source_manager: SourceManager to which selection and filter events will be attached.

    !!! note
        If some results should not be shown, like nan or zero risk, invalid diagnoses,
        etc, remove these data points before passing the data to this class.

    !!! important
        The index of the outer list, is assumed to correspond to the index of the
        individual in source_manager.person_source.
    """

    def __init__(
        self,
        results_per_person: list[list[Hashable]],
        class_list: list[Hashable] | dict[Hashable, str],
        source_manager: SourceManager,
    ):
        self.results_per_person = results_per_person
        self.class_list = class_list
        self.source_manager = source_manager

        self._number_of_individuals = len(results_per_person)

        self.figure = figure(tools=[])
        self.quad = self._set_up_quad()

        # Mistakenly reports no attribute on_change
        self.source_manager.person_source.selected.on_change(  # type: ignore
            "indices", self.get_update_histogram_callback()
        )
        self.source_manager.view.on_change(
            "filter", self.get_update_histogram_callback()
        )

        # Add label from LabelSelectedMixin
        self.register_label()

        self._set_properties()

    @classmethod
    def from_person_field(
        cls,
        source_manager: SourceManager,
        field: str,
        label_mapper: Callable[[Hashable], str] | None = None,
    ) -> "HistogramPlot":
        """Create a HistogramPlot from a field in the person_source.

        Assumes the values of the field are hashable.
        Automatically handles the class list.

        label_mapper: function to map the labels to something more readable
        """
        results_per_person = cast(
            list[list[Hashable]], source_manager.person_source.data[field]
        )
        classes = set(itertools.chain.from_iterable(results_per_person))
        labels: list[Hashable] | dict[Hashable, str] = (
            {class_: label_mapper(class_) for class_ in classes}
            if label_mapper
            else list(classes)
        )
        return cls(results_per_person, labels, source_manager)

    def _set_up_quad(self) -> Quad:
        number_of_classes = len(self.class_list)
        return self.figure.quad(
            top=self.compute_histogram_data(range(self._number_of_individuals)),
            bottom=0,
            left=np.arange(0, number_of_classes) - 0.5,
            right=np.arange(1, number_of_classes + 1) - 0.5,
            fill_color="navy",
            line_color="white",
            alpha=0.5,
            name="quad",
        )

    def _set_properties(self) -> None:
        """Set properties of the figure."""
        tick_labels = (
            self.class_list.values()
            if isinstance(self.class_list, dict)
            else self.class_list
        )
        tick_labels_map = {i: str(label) for i, label in enumerate(tick_labels)}
        properties: dict[str, dict[str, Any]] = {
            "y_range": {"start": 0},
            "xaxis": {
                "axis_label": "State",
                "ticker": list(range(len(self.class_list))),
                "major_label_overrides": tick_labels_map,
                "major_label_orientation": np.pi / 4,
            },
            "yaxis": {"axis_label": "Count"},
            "grid": {"grid_line_color": "white"},
        }

        for module, module_options in properties.items():
            for option, value in module_options.items():
                setattr(getattr(self.figure, module), option, value)

    def compute_histogram_data(self, selected_indices: Iterable[int]) -> list[int]:
        """Compute histogram data from selected indices.

        The selected_indices refer to the indices of the person_source.
        The output is a list of counts for each state, sorted by state. The length is
        the number of states, which is 4."""
        return self._compute_histogram_data(
            selected_indices=selected_indices,
            class_list=self.class_list,
            results_per_person=self.results_per_person,
        )

    @staticmethod
    def _compute_histogram_data(
        selected_indices: Iterable[int],
        class_list: Iterable[Hashable],
        results_per_person: list[list[Hashable]],
    ) -> list[int]:
        state_occurrences = Counter(
            itertools.chain.from_iterable(
                results_per_person[i] for i in selected_indices
            )
        )
        return [state_occurrences[i] for i in class_list]

    def get_update_histogram_callback(self) -> Callable[..., None]:
        """Get a callback function for updating the histogram.

        The callback is attached to multiple quantities, so we do not use
        the supplied old/new values, but use directly the state in source_manager."""

        def update_histogram(attr, old, new) -> None:
            # If nothing is selected, interpret it as everything is selected.
            selected_indices = set(
                self.source_manager.person_source.selected.indices  # type: ignore
                or range(self._number_of_individuals)
            )
            filtered_indices = set(
                parse_filter_to_indices(
                    self.source_manager.view.filter, self._number_of_individuals  # type: ignore
                )
            )
            filtered_and_selected_indices = selected_indices & filtered_indices
            self.quad.data_source.data["top"] = self.compute_histogram_data(
                filtered_and_selected_indices
            )

        return update_histogram


class FilterValueUIElement(Enum):
    RangeSlider = auto()
    """A simple range slider."""
    MultiChoice = auto()
    """Select multiple from a selection."""
    BoolCombination = auto()
    """A composition of all child elements."""
    NoValue = auto()
    """Filters without any value, only on/off."""


FILTER_TO_FilterValueUIElement_MAPPING = {
    BaseFilter: FilterValueUIElement.NoValue,
    CategoricalFilter: FilterValueUIElement.MultiChoice,
    SimpleFilter: FilterValueUIElement.NoValue,
    PersonSimpleFilter: FilterValueUIElement.NoValue,
    ExamSimpleFilter: FilterValueUIElement.NoValue,
    RangeFilter: FilterValueUIElement.RangeSlider,
    BooleanFilter: FilterValueUIElement.BoolCombination,
}


def get_filter_element_from_source_manager(
    filter_name: str,
    source_manager: SourceManager,
    label: str | None = None,
) -> LayoutDOM:
    if filter_name not in source_manager.filters:
        raise ValueError(f"The source manager does not have the filter {filter_name}.")
    filter = source_manager.filters[filter_name]
    return get_filter_element(filter, label or filter_name)


def get_filter_element(filter: BaseFilter, label_text: str = "") -> LayoutDOM:
    """Return a filter element corresponding to a filter in a source_manager."""

    activation_toggle = Switch(active=False)
    inversion_toggle = Switch(active=False)

    activation_toggle.on_change("active", filter.get_set_active_callback())
    inversion_toggle.on_change("active", filter.get_set_inverted_callback())

    match (FILTER_TO_FilterValueUIElement_MAPPING[type(filter)]):
        case FilterValueUIElement.RangeSlider:
            value_element = RangeSlider(value=(0, 100), start=0, end=100, width=None)
            value_element.on_change("value", filter.get_set_value_callback())
        case FilterValueUIElement.MultiChoice:
            # We guarantuee CategoricalFilter inside this match
            filter = cast(CategoricalFilter, filter)
            value_element = MultiChoice(
                value=filter._categories,
                options=list(
                    key for key in filter.category_to_indices.keys() if key is not None
                ),
                stylesheets=[
                    ".choices__list {background-color: #1f2937;} .is-highlighted {background-color: #374151 !important;}"
                ],
            )
            value_element.on_change("value", filter.get_set_value_callback())
        case FilterValueUIElement.BoolCombination:
            value_element = grid(
                column(
                    row(
                        [
                            Paragraph(text="Active"),
                            Paragraph(text="Invert"),
                            Paragraph(text="Value"),
                        ]
                    ),
                    *(
                        get_filter_element(element, label_text=label)
                        for label, element in cast(
                            BooleanFilter, filter
                        ).filters.items()
                    ),
                )
            )
            value_element.stylesheets = [
                ":host {grid-template-rows: unset; grid-template-columns: unset;}"
            ]
        case FilterValueUIElement.NoValue:
            value_element = None
        case _:
            raise ValueError()

    label = Paragraph(text=label_text)

    base_row = cast(Row, row([label, activation_toggle, inversion_toggle]))
    return (
        column(
            base_row,
            value_element,
        )
        if value_element
        else base_row
    )
