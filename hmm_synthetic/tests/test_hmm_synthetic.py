from contextlib import nullcontext

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from hmm_synthetic import data_generator
from hmm_synthetic.backend import sojourn, transition, utils


def test_an_exception() -> None:
    with pytest.raises(ZeroDivisionError):
        5 / 0


def test__simulate_history() -> None:
    age_partitions = np.array(
        [
            (16, 20),
            (20, 25),
            (25, 30),
            (30, 35),
            (35, 40),
            (40, 50),
            (50, 60),
            (60, 100),
        ]
    )
    maximum_age = np.max(age_partitions)
    start_time = 5
    end_time = 60
    number_of_iterations = 10

    for seed in range(number_of_iterations):
        rnd = np.random.default_rng(seed=seed)
        history = data_generator._simulate_history(
            age_min_pts=start_time,
            age_max_pts=end_time,
            time_grid=age_partitions,
            rnd=rnd,
        )

        assert history.shape == (maximum_age,)
        assert np.all(history[:start_time] == 0)
        assert np.all(history[end_time:] == 0)
        assert history[start_time] != 0

    with pytest.raises(ValueError):  # Negative min makes no sense.
        history = data_generator._simulate_history(
            age_min_pts=-1,
            age_max_pts=end_time,
            time_grid=age_partitions,
            rnd=rnd,
        )

    with pytest.raises(ValueError):  # Max should not be larger than the grid.
        history = data_generator._simulate_history(
            age_min_pts=start_time,
            age_max_pts=maximum_age + 1,
            time_grid=age_partitions,
            rnd=rnd,
        )


###################
## Backend tests ##
###################


def test_age_partitions_pts() -> None:
    # Program was written with 4 in mind, so choose 5 to find any assumptions about 4
    points_per_year = 5
    expected_result = np.array(
        [
            [0, 20],
            [20, 45],
            [45, 70],
            [70, 95],
            [95, 120],
            [120, 170],
            [170, 220],
            [220, 420],
        ]
    )

    result = utils.age_partitions_pts(points_per_year)
    assert np.all(result == expected_result)


def test_age_group_idx() -> None:
    """Test the age_group_idx function.

    TODO: There are lots of unhandled edge cases here, which
    we should handle. F.ex. empty partiion list, negative numbers, a number
    that is smaller than the smamllest partiion number...
    """
    age_partitions = [
        [0, 20],
        [20, 45],
        [45, 70],
        [70, 95],
        [95, 120],
        [120, 170],
        [170, 220],
        [220, 420],
    ]

    # Values and correct indices to check
    to_find = [[0, 0], [10, 0], [20, 1], [30, 1], [200, 6], [220, 7], [420, 7]]

    for value, correct_index in to_find:
        computed_index = utils.age_group_idx(value, age_partitions)
        assert computed_index == correct_index

    illegal_ages = [-1, 1000]
    for age in illegal_ages:
        with pytest.raises(ValueError):
            utils.age_group_idx(age, age_partitions)


def test_initial_state() -> None:

    ### Test that the correct state is predicted ###
    age_partitions = np.array(
        [
            [0, 20],
            [20, 45],
            [45, 70],
            [70, 95],
            [95, 120],
            [120, 170],
            [170, 220],
            [220, 420],
        ]
    )
    seed = 42

    result = transition.initial_state(
        30, age_partitions, rnd=np.random.default_rng(seed=seed)
    )
    expected_state = 1
    assert result == expected_state

    ### Test that setting probability to zero for a state makes it never appear ###
    probabilities = np.array(
        [
            [0, 0.5, 0.5],
            [0, 0.5, 0.5],
        ]
    )

    age_partitions = np.array(
        [
            [0, 10],
            [10, 100],
        ]
    )

    rnd = np.random.default_rng(seed=seed)
    number_of_iterations = 10
    for _ in range(number_of_iterations):
        state = transition.initial_state(
            5,
            age_partitions,
            initial_state_probabilities=probabilities,
            rnd=rnd,
        )
        # We set probability of state zero to zero, so should never be produced.
        assert state != 1


def test_legal_transition_lambdas():
    # Array of shape (number_of_age_partitions x number_of_possible_transitions).
    # The transitions are
    # N0->L1  L1->H2   H2->C3   L1->N0   H2->L1   N0->D4   L1->D4   H2->D4   C3->D4
    # correspondingly.
    # I.e. transition_rates[0, 2] is the transition_rate for H2->C3 for the first
    # age partition.
    transition_rates = utils.lambda_sr
    transition_list = [
        [0, 5],
        [3, 1, 6],
        [2, 7, 4],
        [8],
    ]
    age_group_idx = 0  # Chosen arbitrarily
    for state_idx, transitions in enumerate(transition_list):
        state = state_idx + 1  # State is 1-indexed
        correct_transition_rates = [
            transition_rates[age_group_idx, transition] for transition in transitions
        ]
        computed_transition_rates = transition.legal_transition_lambdas(
            state, age_group_idx
        )
        assert np.all(correct_transition_rates == computed_transition_rates)

    illegal_states = [0, 5]
    for state in illegal_states:
        with pytest.raises(ValueError, match="Invalid current state"):
            transition.legal_transition_lambdas(state, age_group_idx)


@given(st.integers())
def test_next_state(current_state: int):
    age_partitions = np.array(
        [
            [0, 20],
            [20, 45],
            [45, 70],
            [70, 95],
            [95, 120],
            [120, 170],
            [170, 220],
            [220, 420],
        ]
    )
    age = 30  # Chosen arbitrarily
    seed = 42
    censoring_value = 0
    rnd = np.random.default_rng(seed=seed)

    max_state = 3
    min_state = 1
    # After this state, next state is always censored.
    automatic_censored_state = max_state + 1
    if not (min_state <= current_state <= max_state):  # Illegal state
        cm = pytest.raises(ValueError, match="Invalid current state")
    else:
        cm = nullcontext()
    if current_state == automatic_censored_state:
        allowed_states = {censoring_value}
    else:
        allowed_states = {
            min(current_state + 1, max_state),
            max(current_state - 1, min_state),
            censoring_value,
        }
    with cm:
        next_state = transition.next_state(
            age_at_exit_pts=age,
            current_state=current_state,
            time_grid=age_partitions,
            rnd=rnd,
            censoring=censoring_value,
        )
        assert next_state in allowed_states


@given(st.integers())
def test_time_exit_state(current_age_pts: int) -> None:
    age_partitions = np.array(
        [
            [0, 20],
            [20, 45],
            [45, 70],
            [70, 95],
            [95, 120],
            [120, 170],
            [170, 220],
            [220, 420],
        ]
    )
    age_max_pts = np.max(age_partitions)
    seed = 42
    state = 2  # Chosen arbitrarily
    rnd = np.random.default_rng(seed=seed)
    exit_time = sojourn.time_exit_state(
        current_age_pts,
        age_max_pts=age_max_pts,
        state=state,
        time_grid=age_partitions,
        rnd=rnd,
    )
    assert exit_time <= age_max_pts
    assert exit_time > 1
