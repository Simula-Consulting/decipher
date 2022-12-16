import dataclasses
import itertools
import types

from hypothesis import given, note
from hypothesis import strategies as st

from bokeh_demo.pilot import (
    ExamTypes,
    Person,
    TimeConverter,
    _combine_dicts,
    _combine_scatter_dicts,
    get_inverse_mapping,
    get_position_list,
)


@given(st.lists(st.integers()))
def test_get_position_list(list_of_numbers: list[int]) -> None:
    position_list = get_position_list(list_of_numbers)

    sorted_list_of_numbers = sorted(list_of_numbers)
    for number, position in zip(list_of_numbers, position_list):
        assert sorted_list_of_numbers[position] == number


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


COARSE_TO_RESULT_MAPPING = get_inverse_mapping()


def exam_result_strategy(coarse_result: int | None = None):
    if coarse_result is None:
        sample_set = itertools.chain.from_iterable(COARSE_TO_RESULT_MAPPING.values())
    else:
        sample_set = COARSE_TO_RESULT_MAPPING[coarse_result]
    return st.sampled_from(sample_set)


@st.composite
def person_strategy(
    draw,
    number_of_states: int = 4,
    number_of_time_steps: int | None = None,
    min_number_exams: int = 3,
):

    if number_of_time_steps is not None and number_of_time_steps <= 0:
        raise ValueError("Number of time steps must be at least 1.")

    number_of_time_steps = number_of_time_steps or draw(
        st.integers(min_value=1, max_value=100)
    )

    # Create exam results
    exam_results = draw(
        st.lists(
            st.sampled_from(range(number_of_states + 1)),
            min_size=number_of_time_steps,
            max_size=number_of_time_steps,
        ).filter(lambda lst: len([el for el in lst if el != 0]) >= min_number_exams)
    )
    detailed_exam_results = [
        draw(exam_result_strategy(state)) if state else None for state in exam_results
    ]
    prediction_time = draw(st.integers(min_value=0, max_value=number_of_time_steps - 1))
    prediction_probabilities = draw(
        st.lists(
            st.floats(min_value=0, max_value=1),
            min_size=number_of_states,
            max_size=number_of_states,
        )
        .filter(lambda p: any(p))  # Not all zero
        .map(lambda p: [pi / sum(p) for pi in p])  # Normalize
    )
    predicted_state = (
        next(
            i
            for i, v in sorted(
                enumerate(prediction_probabilities), key=lambda iv: iv[1], reverse=True
            )
        )
        + 1
    )

    return Person(
        index=0,  # TODO: find out how to do the index
        year_of_birth=draw(st.floats(min_value=1960, max_value=2000)),
        vaccine_age=draw(st.one_of(st.integers(min_value=0, max_value=30), st.none())),
        exam_results=exam_results,
        detailed_exam_results=detailed_exam_results,
        predicted_exam_result=predicted_state,
        prediction_time=prediction_time,
        prediction_probabilities=prediction_probabilities,
    )


def _guarded_length(object, simple_types=(int, float, str)):
    """Return the length of object if it has a length, 1 if it is a simple type."""
    if isinstance(object, simple_types):
        return 1

    try:
        return len(object)
    except TypeError as e:  # Object has no length
        raise ValueError("The object is neither a simple type nor has length.") from e


@given(person_strategy())
def test_person_source_dict(person: Person):
    """Test that the person source dict contains all it is supposed to"""
    source_dict = person.as_source_dict()

    # Check that values are not generators (which are non-serializable)
    for value in source_dict.values():
        assert not isinstance(value, types.GeneratorType)

    for state, detailed_state in zip(
        source_dict["exam_results"], source_dict["detailed_exam_results"]
    ):
        if state == 0:
            assert detailed_state is None
        else:
            # detailed_state is not of type ExamResult, but converted to a dict
            assert isinstance(detailed_state["type"], ExamTypes)
            assert detailed_state in [
                dataclasses.asdict(result) for result in COARSE_TO_RESULT_MAPPING[state]
            ]

    # Each of these must have the same "shape"
    SAME_LENGTH = (
        (
            "exam_results",
            "detailed_exam_results",
            "predicted_exam_results",
            "exam_time_age",
        ),
    )
    for key_set in SAME_LENGTH:
        assert all(key in source_dict for key in key_set)
        assert len({_guarded_length(source_dict[key]) for key in key_set}) == 1

    # Life lines should only have two points
    # For unvaccinated, it should be empty
    has_vaccine = person.vaccine_age is not None
    for key, length in (
        ("lexis_line_endpoints_person_index", 2),
        ("lexis_line_endpoints_age", 2),
        ("lexis_line_endpoints_year", 2),
        ("vaccine_line_endpoints_age", 2 if has_vaccine else 0),
        ("vaccine_line_endpoints_year", 2 if has_vaccine else 0),
    ):
        assert len(source_dict[key]) == length


@given(person_strategy())
def test_person_scatter_source_dict(person: Person):
    """Test that the person scatter source dict contains all it is supposed to"""
    scatter_source_dict = person.as_scatter_source_dict()
    note(scatter_source_dict)

    # Check that values are lists, with elements being int, float, or str
    for value in scatter_source_dict.values():
        assert isinstance(value, list)
        assert all(isinstance(element, int | float | str) for element in value)

    # Only observed values should be included
    assert all(exam != 0 for exam in scatter_source_dict["state"])

    # All lists should be the same length
    assert len({len(value) for value in scatter_source_dict.values()}) == 1

    # Expected keys with type. Use None for any type
    # There may be other keys that in the dict, these are the minimum requirement.
    EXPECTED_KEYS = (
        ("age", float),
        ("year", float),
        ("state", int),
        ("person_index", int),
        ("exam_type", str),
        ("exam_result", str),
    )
    for key, type in EXPECTED_KEYS:
        note(f"key: {key}, type: {type}")
        assert key in scatter_source_dict
        assert all(isinstance(element, type) for element in scatter_source_dict[key])


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
