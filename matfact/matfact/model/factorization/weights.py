from typing import Sequence

import numpy as np
import numpy.typing as npt

from matfact import settings


def data_weights(
    observed_data_matrix: npt.NDArray[np.int_], weights: Sequence[float] | None = None
):
    """Construct a weight matrix for observed data.

    weights contains the weights that should be given to the various states. I.e. the
    first element of weights is the weighing of state 1, the second element for state 2,
    and so on.

    Raises ValueError if there are observed states for which no weight is given.
    """
    if weights is None:
        weights = settings.default_weights

    assert np.min(observed_data_matrix) >= 0  # Observed data should never be negative
    if np.max(observed_data_matrix) > len(weights):
        raise ValueError("The observed data have states for which no weight is given.")
    weight_matrix = np.zeros(observed_data_matrix.shape)
    for i, weight in enumerate(weights):
        state = i + 1  # States are one indexed
        weight_matrix[observed_data_matrix == state] = weight
    return weight_matrix
