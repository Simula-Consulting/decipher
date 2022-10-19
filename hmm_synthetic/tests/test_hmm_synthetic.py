import numpy as np
import pytest

from hmm_synthetic.backend import transition, utils


def test_an_exception() -> None:
    with pytest.raises(ZeroDivisionError):
        5 / 0


def test_initial_state() -> None:
    # Probabilities for having a certain initial state
    # Each row corresponds to an age group, each column to a state.
    initial_state_probabilities = np.array(
        [
            [0.92992, 0.06703, 0.00283, 0.00022],
            [0.92881, 0.06272, 0.00834, 0.00013],
            [0.93408, 0.04945, 0.01632, 0.00015],
            [0.94918, 0.03554, 0.01506, 0.00022],
            [0.95263, 0.03250, 0.01463, 0.00024],
            [0.95551, 0.03303, 0.01131, 0.00015],
            [0.96314, 0.02797, 0.00860, 0.00029],
            [0.96047, 0.02747, 0.01168, 0.00038],
        ]
    )

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
    assert result == 1
