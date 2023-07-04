import itertools
import types

import pytest
from bokeh.models import AllIndices, IndexFilter
from hypothesis import given
from hypothesis import strategies as st

from viz_tool.backend import (
    TimeConverter,
    _combine_dicts,
    _combine_scatter_dicts,
    parse_filter_to_indices,
)
from viz_tool.exam_data import (
    EXAM_RESULT_LOOKUP,
    EXAM_RESULT_MAPPING,
    Diagnosis,
    ExamTypes,
)
from viz_tool.faker import coarse_to_exam_result


@st.composite
def dictionaries_list(
    draw,
    dictionary_element_strategy=st.one_of(st.integers(), st.lists(st.integers())),
    min_number_of_dicts: int = 0,
):
    keys = draw(st.lists(st.sampled_from("abcd"), max_size=5))
    return draw(
        st.lists(
            st.fixed_dictionaries({key: dictionary_element_strategy for key in keys}),
            min_size=min_number_of_dicts,
        )
    )


@given(dictionaries_list())
def test_combine_dicts(dictionaries: list[dict]) -> None:
    combined_dict = _combine_dicts(dictionaries)

    for key, value in combined_dict.items():
        assert value == [dic[key] for dic in dictionaries]


@given(
    dictionaries_list(
        min_number_of_dicts=1, dictionary_element_strategy=st.lists(st.integers())
    )
)
def test_combine_scatter_dicts(dictionaries: list[dict]) -> None:
    combined_dict = _combine_scatter_dicts(dictionaries)

    for key, value in combined_dict.items():
        try:
            flattened = list(
                itertools.chain.from_iterable((dic[key] for dic in dictionaries))
            )
        except TypeError:  # Elements are ints, not lists
            flattened = [dic[key] for dic in dictionaries]
        assert all(isinstance(element, int) for element in flattened)
        assert value == flattened


COARSE_TO_RESULT_MAPPING = coarse_to_exam_result()


def exam_result_strategy(coarse_result: int | None = None):
    if coarse_result is None:
        sample_set = list(
            itertools.chain.from_iterable(COARSE_TO_RESULT_MAPPING.values())
        )
    else:
        sample_set = COARSE_TO_RESULT_MAPPING[coarse_result]
    return st.sampled_from(sample_set)


def _guarded_length(object, simple_types=(int, float, str)):
    """Return the length of object if it has a length, 1 if it is a simple type."""
    if isinstance(object, simple_types):
        return 1

    try:
        return len(object)
    except TypeError as e:  # Object has no length
        raise ValueError("The object is neither a simple type nor has length.") from e


# TODO test source discrepancy. Endpoints match extremal exam points etc


def _get_times(max_time: int = 1000):
    """Return a strategy for either one time point or a list of them."""
    time_point_strategy = st.integers(min_value=0, max_value=max_time)
    return st.one_of(time_point_strategy, st.lists(time_point_strategy))


@given(
    _get_times(),
    st.integers(min_value=0, max_value=24),
    st.integers(min_value=2, max_value=6),
)
def test_time_converter(
    times: int | list[int], zero_point_age: int, points_per_year: int
) -> None:
    """See that time converting to age and back gives the original time."""
    time_converter = TimeConverter(zero_point_age, points_per_year)
    there_and_back = time_converter.age_to_time_point(
        time_converter.time_point_to_age(times)
    )
    if isinstance(times, list):
        assert isinstance(there_and_back, types.GeneratorType)
        assert list(there_and_back) == times
    else:
        assert there_and_back == times


@pytest.mark.parametrize(
    "composite_filter, number_of_indices, result_indices",
    (
        (IndexFilter((1, 2)) | IndexFilter((2, 4)), 10, {1, 2, 4}),
        (
            IndexFilter((1, 2)) | IndexFilter((2, 4)) | IndexFilter((2, 4)),
            10,
            {1, 2, 4},
        ),
        (
            IndexFilter((1, 2)) | IndexFilter((2, 4)) | IndexFilter((2, 9)),
            10,
            {1, 2, 4, 9},
        ),
        (IndexFilter((1, 2)) & IndexFilter((2, 4)), 10, {2}),
        (IndexFilter((1, 2)) & ~IndexFilter((2, 4)), 10, {1}),
        (~IndexFilter((2, 4)), 10, {0, 1, 3, 5, 6, 7, 8, 9}),
        (IndexFilter(()) & IndexFilter(()), 10, set()),
        (AllIndices() & IndexFilter((1, 2)), 10, {1, 2}),
        (
            IndexFilter((1, 2, 7, 6))
            ^ IndexFilter((2, 3, 4, 7))
            ^ IndexFilter((4, 5, 6, 7)),
            8,
            {1, 3, 5, 7},
        ),
        (IndexFilter((1, 2)) ^ ~IndexFilter((2, 3)), 4, {0, 2}),
    ),
)
def test_parse_filter_to_indices(composite_filter, number_of_indices, result_indices):
    """Test the Filter parser."""
    assert (
        parse_filter_to_indices(composite_filter, number_of_indices) == result_indices
    )


def test_parse_filter_raises():
    with pytest.raises(ValueError, match="Parse not implemented for *"):
        parse_filter_to_indices({1, 2}, 10)


def test_diagnosis_lookup():
    """Test that all diagnoses are assigned and mapped."""
    for exam_type in ExamTypes:
        assert exam_type in EXAM_RESULT_LOOKUP, f"{exam_type} not mapped to diagnoses."

    diagnoses_with_mapping = list(
        itertools.chain.from_iterable(EXAM_RESULT_LOOKUP.values())
    )
    for diagnosis in Diagnosis:
        assert (
            diagnosis in diagnoses_with_mapping
        ), f"{diagnosis} is not mapped to by any exam type"
        assert (
            diagnosis in EXAM_RESULT_MAPPING
        ), f"{diagnosis} is not mapped to a coarse state"
