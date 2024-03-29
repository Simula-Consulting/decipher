from contextlib import nullcontext
from typing import ContextManager

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from hmm_synthetic import data_generator
from hmm_synthetic.backend import sojourn, transition, utils
from hmm_synthetic.settings import settings


# The fixture is static, and to play nicely with
# hypothesis, we do not want it to be function-scoped.
# See https://hypothesis.readthedocs.io/en/latest/healthchecks.html#hypothesis.HealthCheck.function_scoped_fixture  # noqa: E501
# for a detailed reference.
# An alternative is to simply suppress the warning from Hypothesis, but we concluded
# it was better to make the scope module. This is safe, as the fixture is completely
# static.
@pytest.fixture(scope="module")
def age_partitions() -> np.ndarray:
    """Return the default age partition array."""
    return np.array([0, 20, 45, 70, 95, 120, 170, 220, 420])


@pytest.fixture
def rnd() -> np.random.Generator:
    """Return a numpy random generator instance."""
    return np.random.default_rng(seed=42)


@given(seed=st.integers(min_value=0))
def test__simulate_history(seed: int, age_partitions: np.ndarray) -> None:
    """Test that the history generated passes some sanity checks."""
    maximum_age = np.max(age_partitions)
    start_time = 5
    end_time = 60

    history = data_generator._simulate_history(
        age_min_pts=start_time,
        age_max_pts=end_time,
        time_grid=age_partitions,
        rnd=np.random.default_rng(seed=seed),
    )

    assert history.shape == (maximum_age,)
    assert np.all(history[:start_time] == 0)  # No values set before start_time
    assert np.all(history[end_time:] == 0)  # No values set after end_time
    # The start time should never be zero.
    # It may happen that times between start and end are zero, a process
    # called censoring.
    assert history[start_time] != 0


maximum_age = 420  # The largest number in age_partitions
# Some arbitrarily chosen values
start_time = 5
end_time = 400


@pytest.mark.parametrize(
    "min, max",
    [
        (-1, end_time),  # Negative start time
        (start_time, maximum_age + 1),  # End time after max age
        (maximum_age, start_time),  # Start time before end time
    ],
)
def test__simulate_history_illegal_min_max_age(
    min: int, max: int, age_partitions: np.ndarray, rnd: np.random.Generator
) -> None:
    """Test that ValueError is raised for illegal min/max ages in history generator."""

    with pytest.raises(ValueError):
        data_generator._simulate_history(
            age_min_pts=min,
            age_max_pts=max,
            time_grid=age_partitions,
            rnd=rnd,
        )


###################
## Backend tests ##
###################


def test_age_partitions_pts(age_partitions: np.ndarray) -> None:
    """Test that the age_partitions are as expected."""
    # Program was written with 4 in mind, so choose 5 to find any assumptions about 4
    points_per_year = 5
    expected_result = age_partitions

    result = utils.age_partitions_pts(points_per_year)
    assert np.all(result == expected_result)


@pytest.mark.parametrize(
    "age, correct_index",
    [[0, 0], [10, 0], [20, 1], [30, 1], [200, 6], [220, 7], [420, 7]],
)
def test_age_group_idx_correct(
    age_partitions: np.ndarray, age: int, correct_index: int
) -> None:
    """Test that the correct age partition index is returned."""
    computed_index = utils.age_group_idx(age, age_partitions)
    assert computed_index == correct_index


@pytest.mark.parametrize("illegal_age", [-1, 1000])
def test_age_group_idx_raises(age_partitions: np.ndarray, illegal_age: int):
    """Test that illegal ages raises ValueError in age_group_idx."""
    with pytest.raises(ValueError):
        utils.age_group_idx(illegal_age, age_partitions)


def test_initial_state(age_partitions: np.ndarray, rnd: np.random.Generator) -> None:
    """Test that the correct initial state is generated.

    TODO: This test is very naive, and should be updated."""
    ### Test that the correct state is predicted ###

    result = transition.initial_state(30, age_partitions, rnd=rnd)
    expected_state = 1
    assert result == expected_state


@given(st.integers(min_value=0), st.integers(min_value=1, max_value=3))
def test_initial_state_probabilities(seed: int, illegal_state: int) -> None:
    """Test that a state of zero probability is never set as initial state."""
    rnd = np.random.default_rng(seed=seed)
    ### Test that setting probability to zero for a state makes it never appear ###
    probabilities = np.full((2, 3), 0.5)
    probabilities[:, illegal_state - 1] = 0

    age_partitions = np.array([0, 10, 100])

    state = transition.initial_state(
        5,  # Arbitrary initial time
        age_partitions,
        initial_state_probabilities=probabilities,
        rnd=rnd,
    )
    # We set probability of state zero to zero, so should never be produced.
    assert state != illegal_state


def test_legal_transition_lambdas() -> None:
    """Test that legal_transition_lambdas returns the correct values."""
    # Array of shape (number_of_age_partitions x number_of_possible_transitions).
    # The transitions are
    # N0->L1  L1->H2   H2->C3   L1->N0   H2->L1   N0->D4   L1->D4   H2->D4   C3->D4
    # correspondingly.
    # I.e. transition_rates[0, 2] is the transition_rate for H2->C3 for the first
    # age partition.
    transition_rates = settings.static.lambda_sr
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


@given(current_state=st.integers(), seed=st.integers(min_value=0))
def test_next_state(current_state: int, seed: int, age_partitions: np.ndarray) -> None:
    """Test that the next state follows the rules for transitioning."""
    age = 30  # Chosen arbitrarily
    censoring_value = 0
    rnd = np.random.default_rng(seed=seed)

    max_state = 4
    min_state = 1
    # After this state, next state is always censored.
    automatic_censored_state = max_state + 1
    cm: ContextManager
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


@pytest.mark.skip(reason="Correct behavior not specified")
@given(current_age_pts=st.integers(), seed=st.integers(min_value=0))
def test_time_exit_state(current_age_pts: int, seed: int, age_partitions) -> None:
    """Test that the generated exit time is legal."""
    age_max_pts = np.max(age_partitions)
    state = 2  # Chosen arbitrarily
    exit_time = sojourn.time_exit_state(
        current_age_pts,
        age_max_pts=age_max_pts,
        state=state,
        time_grid=age_partitions,
        rnd=np.random.default_rng(seed=seed),
    )
    assert exit_time + current_age_pts <= age_max_pts
    assert exit_time > 0


##################
## Test sojourn ##
##################


@pytest.mark.parametrize(
    "random_uniform_variable, age_partition_index, age, current_state, expected_l",
    [
        [0.5, 0, 0, 2, -1],
        [0.5, 0, 10, 2, -1],
        [0.5, 1, 20, 2, 0],
        [0.5, 1, 30, 2, 0],
        [0.5, 6, 200, 2, 5],
        [0.5, 7, 220, 2, 7],
        [0.5, 7, 420, 2, 7],
        #
        [0.2, 0, 0, 2, -1],
        [0.2, 0, 0, 3, -1],
        [0.5, 0, 0, 2, -1],
        [0.5, 0, 0, 3, -1],
        [0.8, 0, 0, 2, -1],
        [0.8, 0, 0, 3, -1],
        [0.2, 1, 20, 2, 0],
        [0.2, 1, 20, 3, 0],
        [0.5, 1, 20, 2, 0],
        [0.5, 1, 20, 3, 0],
        [0.8, 1, 20, 2, 0],
        [0.8, 1, 20, 3, 0],
        [0.2, 6, 200, 2, 5],
        [0.2, 6, 200, 3, 5],
        [0.5, 6, 200, 2, 5],
        [0.5, 6, 200, 3, 5],
        [0.8, 6, 200, 2, 5],
        [0.8, 6, 200, 3, 5],
    ],
)
def test_search_l(
    random_uniform_variable,
    age_partition_index,
    age,
    current_state,
    expected_l,
    age_partitions,
):
    assert expected_l == sojourn.search_l(
        random_uniform_variable, age_partition_index, age, current_state, age_partitions
    )


@pytest.mark.parametrize(
    "random_uniform_variable, age, current_state, age_partition_index, l_partition_index, expected_time",
    [
        [0.8, 10, 3, 0, 0, 6.569402475342262],
        [0.2, 20, 2, 1, 6, 22.543897012688554],
        [0.2, 220, 3, 7, 0, 0.9108271819837941],
        [0.8, 10, 2, 0, 0, 8.28326254469429],
        [0.5, 220, 2, 7, 5, 3.4401071048684564],
        [0.5, 220, 3, 7, 6, 6.292185734930512],
        [0.8, 20, 3, 1, 1, 13.951438214581314],
        [0.5, 10, 3, 0, 1, -5.228439835645411],
        [0.8, 220, 2, 7, 1, 8.88014738707846],
        [0.5, 10, 3, 0, 6, 24.619618559912382],
    ],
)
def test_exit_time(
    random_uniform_variable,
    age_partition_index,
    age,
    current_state,
    l_partition_index,
    expected_time,
    age_partitions,
):
    assert expected_time == sojourn.exit_time(
        random_uniform_variable,
        age,
        current_state,
        age_partition_index,
        l_partition_index,
        age_partitions,
    )


@pytest.mark.parametrize(
    "start_age, end_age, seed, points_pr_year, correct_history",
    # fmt: off
    [
        [
            10, 100, 2, 2,
            [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # noqa: E231
        ],
        [
            10, 70, 4, 1,
            [0,0,0,0,0,0,0,0,0,0,2,2,2,2,3,2,2,2,3,3,3,2,2,2,2,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,3,3,3,3,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # noqa: E231
        ],
    ]
    # fmt: on
)
def test_history_matches(start_age, end_age, seed, points_pr_year, correct_history):
    """Assert that the generated history matches correct_history."""
    time_grid = utils.age_partitions_pts(points_pr_year)
    rng = np.random.default_rng(seed)
    history = data_generator._simulate_history(start_age, end_age, time_grid, rng)
    assert np.all(history == np.array(correct_history))
