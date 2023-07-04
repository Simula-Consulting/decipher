from collections import namedtuple

import pytest
from hypothesis import given
from hypothesis import strategies as st

from viz_tool.frontend import HistogramPlot, get_position_list


@given(st.lists(st.integers()))
def test_get_position_list(list_of_numbers: list[int]) -> None:
    position_list = get_position_list(list_of_numbers)

    sorted_list_of_numbers = sorted(list_of_numbers)
    for number, position in zip(list_of_numbers, position_list):
        assert sorted_list_of_numbers[position] == number


histogram_test_data = namedtuple(
    "histogram_test_data",
    ["selected_indices", "class_list", "results_per_person", "expected"],
)
"""Test data for histogram plot."""


@pytest.mark.parametrize(
    "test_case",
    [
        histogram_test_data([0], [1, 2, 3, 4], [[1, 3], [2, 4]], [1, 0, 1, 0]),
        histogram_test_data([1], [1, 2, 3, 4], [[1, 3], [2, 4]], [0, 1, 0, 1]),
        histogram_test_data([0], [3, 4], [[1, 3], [2, 4]], [1, 0]),
        histogram_test_data([0, 1], [1, 2, 3, 4], [[1, 3], [2, 4]], [1, 1, 1, 1]),
        histogram_test_data(
            [0], ["fish", "no-fish", "dog"], [["dog", "dog"], ["fish"]], [0, 0, 2]
        ),
    ],
)
def test_histogram_plot(test_case: histogram_test_data) -> None:
    result = HistogramPlot._compute_histogram_data(
        selected_indices=test_case.selected_indices,
        class_list=test_case.class_list,
        results_per_person=test_case.results_per_person,
    )
    assert result == test_case.expected
