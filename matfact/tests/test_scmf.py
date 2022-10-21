import warnings

import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays

from matfact.experiments import SCMF
from matfact.experiments.algorithms.factorization.scmf import (
    _custom_roll,
    _take_per_row_strided,
)
from matfact.settings import TEST_PATH

artifact_path = TEST_PATH / "test_artifacts" / "SCMF_test"


@given(st.data())
def test_custom_roll(data):
    """Compare _custom_roll with naive slow rolling"""
    array = data.draw(arrays(np.float, array_shapes(min_dims=2, max_dims=2)))
    assume(not np.isnan(array).any())
    # Limit shifts to not be too large (1e4 arbitrarily chosen), as _custom_roll
    # is suseptible to floating point errors for large shifts.
    # This is not relevant for us, as shifts are never larger than the number
    # of time steps.
    shifts = data.draw(
        arrays(
            int,
            array.shape[0],
            elements=st.integers(min_value=-1e4, max_value=1e4),
        )
    )
    rolled = _custom_roll(array, shifts)
    for row, rolled_row, shift in zip(array, rolled, shifts):
        assert np.all(rolled_row == np.roll(row, shift))


@given(st.data())
def test_take_per_row_strided(data):
    A = data.draw(
        arrays(
            float,
            array_shapes(min_dims=2, max_dims=2, min_side=2),
            elements=st.floats(allow_nan=False),
        )
    )
    n_elem = data.draw(st.integers(min_value=0, max_value=A.shape[1] - 1))
    start_idx = data.draw(
        arrays(
            int,
            A.shape[0],
            elements=st.integers(min_value=0, max_value=A.shape[1] - n_elem - 1),
        )
    )
    strided_A = _take_per_row_strided(A, start_idx, n_elem)
    for i, row in enumerate(strided_A):
        assert np.array_equal(row, A[i, start_idx[i] : start_idx[i] + n_elem])


def _generate_SCMF_logs() -> dict[str, np.ndarray]:
    """Helper function to generate quantities from SCMF to be compared."""
    # Parameters used in the model. NB! Do not change unless also regenerating
    # the "truth" artifacts.
    N = 100
    T = 40
    r = 5
    iterations = 4  # Number of iterations to run the solver

    rnd = np.random.default_rng(seed=42)
    X = rnd.integers(low=0, high=4, size=(N, T))  # Initial observation matrix
    V = rnd.random((T, r))  # Initial basic profiles
    s_budget = np.arange(-10, 11)

    # Allocate space for the logs
    logs = {
        "X": np.empty((iterations, *X.shape)),
        # Internally, SCMF pads V with twice the shift length
        # NB: in future implementaions of SCMF we might choose to not
        # do this, in that case this test will fail.
        # If so, either disable checking of V, or
        # figure out the transformation from padded V to actual V.
        "V_bc": np.empty((iterations, V.shape[0] + 2 * len(s_budget), V.shape[1])),
        "M": np.empty((iterations, N, T)),
        "U": np.empty((iterations, N, r)),
        "loss": np.empty(iterations),
        "s": np.empty((iterations, N)),
    }

    scmf = SCMF(X, V, s_budget)

    for i in range(iterations):
        scmf.run_step()
        for attribute in logs:
            attribute_value = getattr(scmf, attribute)
            logs[attribute][i] = (
                attribute_value() if callable(attribute_value) else attribute_value
            )

    return logs


def test_scmf():
    """Snapshot test that SCMF behaves as expected, comparing to stored correct values.

    Rationale:
    We have run the SCMF factorizer and stored its internal matrices and other data
    as function of iteration step as artifacts. This test runs SCMF and compares
    that the values match.

    Snapshots are updated by running the script `generate_scmf_artifacts.py`.
    """

    logs = _generate_SCMF_logs()

    for attribute in logs:
        correct = np.load(artifact_path / f"{attribute}_log.npy")
        observed = logs[attribute]
        # Use allclose instead of array_equal, to allow for refactoring
        # that cuases different round off (for example avoiding sqrt).
        assert np.allclose(correct, observed)
        if not np.array_equal(correct, observed):
            warnings.warn(
                "Test successful, but note that arrays are only similar, "
                "not equal. Consider updating the snapshot."
            )
