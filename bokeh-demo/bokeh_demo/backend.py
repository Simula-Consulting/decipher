from __future__ import annotations  # Postponed evaluation of types

import functools
import itertools
import operator
from collections import defaultdict
from collections.abc import Callable, Container, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar, cast, overload

from bokeh.models import AllIndices, CDSView, ColumnDataSource, CustomJS
from bokeh.models import Filter as BokehFilter
from bokeh.models import (
    IndexFilter,
    IntersectionFilter,
    InversionFilter,
    SymmetricDifferenceFilter,
    UnionFilter,
)


## Time Converter ##
@dataclass
class TimeConverter:
    """Convert between time point index and age."""

    zero_point_age: int = 16
    """Zero point of the measurement data."""
    points_per_year: float = 4
    """Number of time points per year."""

    @overload
    def time_point_to_age(self, time_points: int) -> float:
        ...

    @overload
    def time_point_to_age(self, time_points: Sequence[int]) -> Sequence[float]:
        ...

    def time_point_to_age(self, time_points):
        """Convert time point or points to age."""

        def convert(time):
            return self.zero_point_age + time / self.points_per_year

        try:
            return (convert(time_point) for time_point in time_points)
        except TypeError:  # Only one point
            return convert(time_points)

    @overload
    def age_to_time_point(self, ages: float) -> int:
        ...

    @overload
    def age_to_time_point(self, ages: Sequence[float]) -> Sequence[int]:
        ...

    def age_to_time_point(self, ages):
        """Convert ages to closest time points."""

        def convert(age):
            return round((age - self.zero_point_age) * self.points_per_year)

        try:
            return (convert(age) for age in ages)
        except TypeError:  # Only one point
            return convert(ages)


time_converter = TimeConverter()

## Data structures


def _get_endpoint_indices(history: Sequence[int]) -> tuple[int, int]:
    """Return the first and last index of non-zero entries.

    >>> _get_endpoint_indices((0, 1, 0, 2, 0))
    (1, 3)
    >>> _get_endpoint_indices((0, 1))
    (1, 1)
    """

    def first_nonzero_index(seq):
        return next(i for i, y in enumerate(seq) if y != 0)

    first = first_nonzero_index(history)
    last = len(history) - 1 - first_nonzero_index(reversed(history))
    return first, last


def _calculate_delta(
    probabilities: Sequence[float],
    correct_index: int,
) -> float:
    """Calculate the delta value from probabilities for different classes.

    Note:
        The correct_index is the _index_ of the correct class, not its label!
    """
    incorrect_estimates = (
        *probabilities[:correct_index],
        *probabilities[correct_index + 1 :],
    )
    # Set default=0 for the edge case that there is only one state, in which
    # case incorrect_estimates is empty.
    return max(incorrect_estimates, default=0) - probabilities[correct_index]


def _combine_dicts(dictionaries: Iterable[dict]) -> dict:
    """Combine dictionaries by making lists of observed values.

    >>> a = {'a': 4}
    >>> b = {'a': 3}
    >>> _combine_dicts((a, b))
    {'a': [4, 3]}
    """
    new_dict = defaultdict(list)
    for dictionary in dictionaries:
        for key, value in dictionary.items():
            new_dict[key].append(value)

    return new_dict


def _combine_scatter_dicts(dictionaries: Sequence[dict]) -> dict:
    """Combine dictionaries by making flattened lists of observed values.

    TODO should be combined with the above"""
    dictionary_keys = dictionaries[0].keys()
    assert {key for dic in dictionaries for key in dic.keys()} == set(
        dictionary_keys
    ), "All dictionaries must have the same fields"

    return {
        key: [value for dic in dictionaries for value in dic[key]]
        for key in dictionary_keys
    }


def link_sources(
    person_source: ColumnDataSource, exam_source: ColumnDataSource
) -> None:
    def find_and_set_indices(selected_people):
        exam_indices = [
            exam_inds
            for exam_inds, pid in zip(
                person_source.data["exam_idx"], person_source.data["PID"]
            )
            if pid in selected_people
        ]
        exam_indices = [item for sublist in exam_indices for item in sublist]

        exam_source.selected.indices = exam_indices
        person_source.selected.indices = [
            i
            for i, pid in enumerate(person_source.data["PID"])
            if pid in selected_people
        ]

    def exam_selector_callback(attr, old, new):
        if new == []:  # Avoid unsetting when hitting a line in scatter plot
            return
        selected_people = list({exam_source.data["PID"][i] for i in new})
        find_and_set_indices(selected_people)

    def person_selector_callback(attr, old, new):
        if new == []:  # Avoid unsetting when hitting a line in scatter plot
            return
        selected_people = list({person_source.data["PID"][i] for i in new})
        find_and_set_indices(selected_people)

    exam_source.selected.on_change("indices", exam_selector_callback)  # type: ignore
    person_source.selected.on_change("indices", person_selector_callback)  # type: ignore


@dataclass
class BaseFilter:
    """Filter used by SourceManager to handle CDSView filtering.

    Note:
        This must not be confused by the Filter (imported as BokehFilter) class from Bokeh.
        This class wraps around the Bokeh Filter, and ultimately serves Bokeh Filters
        through `get_filter` and `get_exam_filter`, however, they are not interchangeable.

    !!! tip "See also"
        See [bokeh_demo.frontend.get_filter_element][] on how callbacks may be used.
    """

    source_manager: SourceManager
    active: bool = False
    inverted: bool = False

    def get_set_active_callback(self) -> Callable[[str, bool, bool], None]:
        """Return a callback function for activating the filter."""

        def set_active(attr: str, old: bool, new: bool) -> None:
            self.active = new
            self.source_manager.update_views()

        return set_active

    def get_set_inverted_callback(self) -> Callable[[str, bool, bool], None]:
        """Return a callback function for inverting the filter."""

        def set_inverted(attr: str, old: bool, new: bool) -> None:
            self.inverted = new
            self.source_manager.update_views()

        return set_inverted

    def get_set_value_callback(self) -> Callable[[str, Any, Any], None]:
        """Return a callback function for setting the value of the filter.

        What the value is depends on the filter. Some filters only have on/off, in
        which case there is no value, while others have a value, for example a floating
        point threshold."""
        raise NotImplementedError("abstract")

    def get_filter(self) -> BokehFilter:
        """Get the resulting Filter for the people."""
        raise NotImplementedError("abstract")

    def get_exam_filter(self) -> BokehFilter:
        """Get the resulting Filter for the exams."""
        raise NotImplementedError("abstract")

    @staticmethod
    def _person_to_exam_indices(
        person_indices: Container[int], exam_to_person_mapping: Iterable[int]
    ) -> list[int]:
        """Find all exam indices belonging to the people in person_indices.

        exam_to_person_mapping : a sequence with the person index of each exam, i.e.
            element number n contains the index of the person belonging to exam n.
            Typically, this is retrieved from a source as `exam_source.data["PID"]`
        """
        return [
            i
            for i, person_index in enumerate(exam_to_person_mapping)
            if person_index in person_indices
        ]

    @staticmethod
    def _exam_to_person_indices(
        exam_indices: Iterable[int], exam_to_person_mapping: Sequence[int]
    ) -> list[int]:
        """Given an iterable of exam indices, return the corresponding person indices.

        exam_to_person_mapping : a sequence with the person index of each exam, i.e.
            element number n contains the index of the person belonging to exam n.
            Typically, this is retrieved from a source as `exam_source.data["PID"]`
        """
        return list({exam_to_person_mapping[i] for i in exam_indices})


class SimpleFilter(BaseFilter):
    """Simple index based filter."""

    def __init__(
        self,
        person_indices: Sequence[int],
        exam_indices: Sequence[int],
        source_manager: SourceManager,
        active: bool = False,
        inverted: bool = False,
    ) -> None:
        super().__init__(
            source_manager=source_manager, active=active, inverted=inverted
        )
        self.person_indices = person_indices
        self.exam_indices = exam_indices
        self.person_filter = IndexFilter(self.person_indices)
        self.exam_filter = IndexFilter(self.exam_indices)

    def get_filter(self) -> BokehFilter:
        if not self.active:
            return AllIndices()
        return ~self.person_filter if self.inverted else self.person_filter

    def get_exam_filter(self) -> BokehFilter:
        if not self.active:
            return AllIndices()
        return ~self.exam_filter if self.inverted else self.exam_filter


class PersonSimpleFilter(SimpleFilter):
    """Filter using person indices, exam entries for the people selected also shown."""

    def __init__(
        self,
        person_indices: Sequence[int],
        source_manager: SourceManager,
        active: bool = False,
        inverted: bool = False,
    ) -> None:
        exam_to_person = cast(Sequence[int], source_manager.exam_source.data["PID"])
        exam_indices = self._person_to_exam_indices(person_indices, exam_to_person)
        super().__init__(
            person_indices,
            exam_indices,
            source_manager=source_manager,
            active=active,
            inverted=inverted,
        )


class ExamSimpleFilter(SimpleFilter):
    """Filter using exam indices, people of the entries also shown."""

    def __init__(
        self,
        exam_indices: Sequence[int],
        source_manager: SourceManager,
        active: bool = False,
        inverted: bool = False,
    ) -> None:
        exam_to_person = cast(Sequence[int], source_manager.exam_source.data["PID"])
        person_indices = self._exam_to_person_indices(exam_indices, exam_to_person)
        super().__init__(
            person_indices,
            exam_indices,
            source_manager=source_manager,
            active=active,
            inverted=inverted,
        )


class RangeFilter(BaseFilter):
    """Generic range filter, accepting all people with `field` value within a range.

    Warning:
        The field must contain a numeric value!"""

    _range = (-float("inf"), float("inf"))  # 'Accept all'

    def __init__(
        self,
        field: str,
        source_manager: SourceManager,
        active: bool = False,
        inverted: bool = False,
    ) -> None:
        super().__init__(
            source_manager=source_manager, active=active, inverted=inverted
        )
        self.field = field
        self.selected = self._get_selection_indices()
        self.exams_selected = self._person_to_exam_indices(
            self.selected, self.source_manager.exam_source.data["PID"]
        )

    def _get_selection_indices(self) -> list[int]:
        data = self.source_manager.person_source.data[self.field]
        min, max = self._range
        return [
            i
            for i, value in enumerate(data)
            if value is not None and min <= value <= max
        ]

    def get_set_value_callback(
        self,
    ) -> Callable[[str, tuple[float, float], tuple[float, float]], None]:
        def set_value_callback(
            attr: str, old: tuple[float, float], new: tuple[float, float]
        ) -> None:
            assert len(new) == 2
            self._range = new
            self.selected = self._get_selection_indices()
            self.exams_selected = self._person_to_exam_indices(
                self.selected, self.source_manager.exam_source.data["PID"]
            )
            self.source_manager.update_views()

        return set_value_callback

    def get_filter(self) -> BokehFilter:
        if not self.active:
            return AllIndices()
        return (
            ~IndexFilter(self.selected) if self.inverted else IndexFilter(self.selected)
        )

    def get_exam_filter(self) -> BokehFilter:
        if not self.active:
            return AllIndices()
        return (
            ~IndexFilter(self.exams_selected)
            if self.inverted
            else IndexFilter(self.exams_selected)
        )


T = TypeVar("T")


def _list_reverse_lookup(input_list: Iterable[T]) -> dict[T, list[int]]:
    """Given an iterable, return a dict mapping values to indices.

    Example:
        >>> _list_reverse_lookup((0, 1, 0))
        {0: [0, 2], 1: [1]}
        >>> _list_reverse_lookup("haha")
        {"h": [0, 2], "a": [1, 3]}
    """
    mapping = defaultdict(list)
    for key, value in enumerate(input_list):
        mapping[value].append(key)
    return mapping


class CategoricalFilter(BaseFilter):
    """Filter selecting by categorical data."""

    _categories: list = []
    """The selected categories."""

    def __init__(
        self,
        field: str,
        source_manager: SourceManager,
        active: bool = False,
        inverted: bool = False,
    ) -> None:
        super().__init__(
            source_manager=source_manager, active=active, inverted=inverted
        )
        self.field = field
        self.category_to_indices = _list_reverse_lookup(
            self.source_manager.person_source.data[self.field]
        )

        self.selected = self._get_selection_indices()
        self.exams_selected = self._person_to_exam_indices(
            self.selected, self.source_manager.exam_source.data["PID"]
        )

    def _get_selection_indices(self) -> list[int]:
        return list(
            itertools.chain.from_iterable(
                self.category_to_indices[category] for category in self._categories
            )
        )

    def get_set_value_callback(
        self,
    ) -> Callable[[str, list, list], None]:
        def set_value_callback(attr: str, old: list, new: list) -> None:
            self._categories = new
            self.selected = self._get_selection_indices()
            self.exams_selected = self._person_to_exam_indices(
                self.selected, self.source_manager.exam_source.data["PID"]
            )
            self.source_manager.update_views()

        return set_value_callback

    def get_filter(self) -> BokehFilter:
        if not self.active:
            return AllIndices()
        return (
            ~IndexFilter(self.selected) if self.inverted else IndexFilter(self.selected)
        )

    def get_exam_filter(self) -> BokehFilter:
        if not self.active:
            return AllIndices()
        return (
            ~IndexFilter(self.exams_selected)
            if self.inverted
            else IndexFilter(self.exams_selected)
        )


class BooleanFilter(BaseFilter):
    """Filter combining multiple filters through some logical operation."""

    def __init__(
        self,
        filters: dict[str, BaseFilter],
        source_manager: SourceManager,
        bokeh_bool_filter: type[BokehFilter] = UnionFilter,
        active: bool = False,
        inverted: bool = False,
    ):
        super().__init__(
            source_manager=source_manager, active=active, inverted=inverted
        )
        self.filters = filters
        self.bokeh_bool = bokeh_bool_filter

    def get_filter(self) -> BokehFilter:
        if not self.active:
            return AllIndices()
        filter = self.bokeh_bool(
            operands=[
                filter.get_filter() for filter in self.filters.values() if filter.active
            ]
            or [IndexFilter([])]
        )
        return ~filter if self.inverted else filter

    def get_exam_filter(self) -> BokehFilter:
        if not self.active:
            return AllIndices()
        filter = self.bokeh_bool(
            operands=[
                filter.get_exam_filter()
                for filter in self.filters.values()
                if filter.active
            ]
            or [IndexFilter([])]
        )
        return ~filter if self.inverted else filter


@functools.singledispatch
def parse_filter_to_indices(filter: BokehFilter, number_of_indices: int) -> set[int]:
    """Given a Filter, return the resulting index list.

    Example:
        >>> combined_filter = UnionFilter(IndexFilter(1, 2), IndexFilter(2, 4))
        >>> parse_filter_to_indices(combined_filter, 10)
        (1, 2, 4)

        >>> combined_filter = InversionFilter(IndexFilter(1, 2))
        >>> parse_filter_to_indices(combined_filter, 5)
        (0, 3, 4)
    """

    raise ValueError(f"Parse not implemented for filter type {type(filter)}")


@parse_filter_to_indices.register
def _(filter: IndexFilter, number_of_indices: int) -> set[int]:
    return set(filter.indices)  # type: ignore  # indices has type Nullable from Bokeh


@parse_filter_to_indices.register
def _(filter: AllIndices, number_of_indices: int) -> set[int]:
    return set(range(number_of_indices))


@parse_filter_to_indices.register
def _(filter: IntersectionFilter, number_of_indices: int) -> set[int]:
    return functools.reduce(
        operator.and_,
        (
            set(parse_filter_to_indices(operand, number_of_indices))
            for operand in filter.operands  # type: ignore  # operands has type Required(Seq(Int)) from Bokeh
        ),
    )


@parse_filter_to_indices.register
def _(filter: InversionFilter, number_of_indices: int) -> set[int]:
    return set(range(number_of_indices)) - parse_filter_to_indices(
        filter.operand, number_of_indices  # type: ignore
    )


@parse_filter_to_indices.register
def _(filter: UnionFilter, number_of_indices: int) -> set[int]:
    return functools.reduce(
        operator.or_,
        (
            set(parse_filter_to_indices(operand, number_of_indices))
            for operand in filter.operands  # type: ignore
        ),
    )


@parse_filter_to_indices.register
def _(filter: SymmetricDifferenceFilter, number_of_indices: int) -> set[int]:
    return functools.reduce(
        lambda x, y: x.symmetric_difference(y),
        (
            set(parse_filter_to_indices(operand, number_of_indices))
            for operand in filter.operands  # type: ignore
        ),
    )


class SourceManager:
    def __init__(self, person_source: ColumnDataSource, exam_source: ColumnDataSource):
        self.person_source = person_source
        self.exam_source = exam_source
        link_sources(self.person_source, self.exam_source)

        self.only_selected_view = CDSView(filter=IndexFilter())
        """View for selected people.

        Note:
            You probably don't want to use this view, but rather `combined_view`."""
        # There is apparently some issues in Bokeh with re-rendering on updating
        # filters. See #7273 in Bokeh
        # https://github.com/bokeh/bokeh/issues/7273
        # The emit seems to resolve this for us, but it is rather hacky.
        self.person_source.selected.js_on_change(  # type: ignore  # Wrongly says Readonly selected has no attribute js_on_change
            "indices",
            CustomJS(
                args={"source": self.person_source, "view": self.only_selected_view},
                code="""
            if (source.selected.indices.length){
                view.filter.indices = source.selected.indices;
            } else {
                view.filter.indices = [...Array(source.get_length()).keys()];
            }
            source.change.emit();
            """,
            ),
        )

        self.filters: dict[str, BaseFilter] = {}
        """Filters registered with the source manager.

        The manager's views are updated with filters by calling `update_views`."""

        self.view = CDSView()
        """View for filtered people."""
        self.exam_view = CDSView()
        """View for filtered exams."""
        self.combined_view = CDSView(
            # The view's filters are typed as Instance[Filter] and thus mypy
            # reports unsupported type for the & operator. However, it is
            # supported for Filter, which they actually are.
            filter=self.view.filter
            & self.only_selected_view.filter  # type: ignore
        )
        """View for the intersection of filtered and selected people."""

    def update_views(self) -> None:
        """Set the view's filters to match source_manager's internal filters."""

        # Ideally, we would set self.view's filter to be an IntersectionFilter,
        # initially with the operands [AllIndices()]. Then, we would in `update_views`
        # simply update the operands, but keep the filter object. That way, we would
        # not have to update combined_views's filter.
        # However, updating only the operands, does not trigger re-rendering, as
        # updating attributes of filters is broken.
        # See https://github.com/bokeh/bokeh/issues/7273.
        #
        # It is not possible to first set self.view's filter to the intersection filter,
        # and then update combined_view, as before we have updated combined_view,
        # there is an undefined reference to the old view's filter.
        # Thus, we must to this intermediate step, first setting combined_view, then
        # the main view.
        # This is quite fragile.
        active_person_filters = IntersectionFilter(
            operands=[
                filter.get_filter() for filter in self.filters.values() if filter.active
            ]
            or [AllIndices()]
        )
        self.combined_view.filter = (
            active_person_filters & self.only_selected_view.filter  # type: ignore
        )
        self.view.filter = active_person_filters  # type: ignore
        self.exam_view.filter = IntersectionFilter(  # type: ignore
            operands=[
                filter.get_exam_filter()
                for filter in self.filters.values()
                if filter.active
            ]
            or [AllIndices()]
        )

    def get_vaccine_range(self) -> tuple[float, float]:
        """Return min and max age of vaccine administration."""
        vaccine_ages = [
            age for age in self.person_source.data["vaccine_age"] if age is not None
        ]
        return (min(vaccine_ages), max(vaccine_ages))
