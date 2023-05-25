from hypothesis import given
from hypothesis import strategies as st

from bokeh_demo.frontend import get_position_list


@given(st.lists(st.integers()))
def test_get_position_list(list_of_numbers: list[int]) -> None:
    position_list = get_position_list(list_of_numbers)

    sorted_list_of_numbers = sorted(list_of_numbers)
    for number, position in zip(list_of_numbers, position_list):
        assert sorted_list_of_numbers[position] == number
