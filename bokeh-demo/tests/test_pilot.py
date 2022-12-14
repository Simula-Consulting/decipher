import itertools

from hypothesis import given, strategies as st

from bokeh_demo.pilot import get_position_list, _combine_dicts, _combine_scatter_dicts, Person

@given(st.lists(st.integers()))
def test_get_position_list(list_of_numbers: list[int]) -> None:
    position_list = get_position_list(list_of_numbers)

    sorted_list_of_numbers = sorted(list_of_numbers)
    for number, position in zip(list_of_numbers, position_list):
        assert sorted_list_of_numbers[position] == number


@st.composite
def dictionaries_list(draw, min_number_of_dicts: int = 0):
    keys = draw(st.lists(st.sampled_from('abcd'), max_size=5))
    return draw(st.lists(
        st.fixed_dictionaries({key: st.one_of(st.integers(), st.lists(st.integers())) for key in keys}),
        min_size=min_number_of_dicts,
    ))

@given(dictionaries_list())
def test_combine_dicts(dictionaries: list[dict]) -> None:
    combined_dict = _combine_dicts(dictionaries)

    for key, value in combined_dict.items():
        assert value == [dic[key] for dic in dictionaries]


@given(dictionaries_list(min_number_of_dicts=1))
def test_combine_scatter_dicts(dictionaries: list[dict]) -> None:
    combined_dict = _combine_scatter_dicts(dictionaries)

    for key, value in combined_dict.items():
        try:
            flattened = list(itertools.chain.from_iterable((dic[key] for dic in dictionaries)))
        except TypeError:  # Elements are ints, not lists
            flattened = [dic[key] for dic in dictionaries]
        assert all(isinstance(element, int) for element in flattened)
        assert value == flattened

    
@st.composite
def person_strategy(draw, number_of_states: int = 4, number_of_time_steps: int | None = None):

    if number_of_time_steps is not None and number_of_time_steps <= 0:
        raise ValueError("Number of time steps must be at least 1.")

    number_of_time_steps = number_of_time_steps or draw(st.integers(min_value=1))

    # Create exam results
    exam_results = draw(st.lists(
        st.sampled_from(range(number_of_states + 1)),
        min_size=number_of_time_steps,
        max_size=number_of_time_steps,
        ))
    prediction_time = draw(st.integers(min_value=0, max_value=number_of_states - 1))
    prediction_probabilities = draw(st.lists(
        st.floats(min_value=0, max_value=1),
        min_size=number_of_states,
        max_size=number_of_states,
        )
        .filter(lambda p: any(p))  # Not all zero
        .map(lambda p: [pi / sum(p) for pi in p])  # Normalize
        )
    predicted_state = next(i for i, v in sorted(enumerate(prediction_probabilities), key=lambda iv: iv[1], reverse=True)) + 1
    
    

    Person(
        year_of_birth=st.floats(min_value=1960, max_value=2000),
        exam_results=exam_results,
        predicted_exam_result=predicted_state,
        prediction_time=prediction_time,
        prediction_probabilities=prediction_probabilities,
        lexis_line_endpoints_age=
        lexis_line_endpoints_year=
    )
