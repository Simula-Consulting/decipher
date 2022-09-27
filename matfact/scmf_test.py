import numpy as np

from matfact.experiments import SCMF
from matfact.settings import BASE_PATH

artifact_path = BASE_PATH / "test_artifacts" / "SCMF_test"


def test_scmf():
    """Test that SCMF behaves as expected, comparing to stored correct values.

    Rationale:
    We have run the SCMF factorizer and stored its internal matrices and other data
    as function of iteration step as artifacts. This test runs SCMF and compares
    that the values match.
    """

    # Parameters used in the model. NB! Do not change unless also regenerating
    # the "truth" artifacts.
    N = 100
    T = 40
    r = 5
    iterations = 4  # Number of iterations to run the solver

    np.random.seed(42)
    X = np.random.random((N, T))  # Inital observation matrix
    V = np.random.random((T, r))  # Inital basic profiles
    s_budget = np.arange(-10, 11)

    # Allocate space for the logs
    logs = {
        "X": np.empty((iterations, *X.shape)),
        # Internally, SCMF pads V with twice the shift length
        # NB: in future implementaions of SCMF we might choose to not
        # do this, in that case this test will fail.
        # If so, either disable checking of V, or
        # figure out the transformation from padded V to actual V.
        "V": np.empty((iterations, V.shape[0] + 2 * len(s_budget), V.shape[1])),
        "M": np.empty((iterations, N, T)),
        "U": np.empty((iterations, N, r)),
        "loss": np.empty(iterations),
        "s": np.empty((iterations, N)),
    }

    # Assumes there to exist an array <attribute_name>_log of the appropriate size
    attributes_to_log = ["X", "M", "U", "V", "loss", "s"]

    scmf = SCMF(X, V, s_budget)

    for i in range(iterations):
        scmf.run_step()
        for attribute in attributes_to_log:
            attribute_value = getattr(scmf, attribute)
            logs[attribute][i] = (
                attribute_value() if callable(attribute_value) else attribute_value
            )

    # Uncommenct /only/ to regenerate artifacts!!
    # for attribute in attributes_to_log:
    #     np.save(artifact_path / f"{attribute}_log.npy", logs[attribute])

    for attribute in attributes_to_log:
        correct = np.load(artifact_path / f"{attribute}_log.npy")
        observed = logs[attribute]
        assert np.array_equal(correct, observed)
