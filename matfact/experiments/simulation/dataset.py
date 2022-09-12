import numpy as np


def prediction_data(Y, method):
    """Simulate data for a prediction task based on longitudinal
    vectors in Y. The procedure will select one data point per row in
    Y and remove this from the output matrix. The original data points
    and the corresponding column number (aka time points) are stored in
    separate vectors."""

    return mask_test_samples(Y, _t_pred(Y, method))


def _t_pred(Y, prediction_rule):

    # Find time of last observed entry for all rows
    if prediction_rule == "last_observed":
        return Y.shape[1] - np.argmax(Y[:, ::-1] != 0, axis=1) - 1

    if prediction_rule == "random":

        rows, cols = np.where(Y != 0)

        times = []
        for r in np.unique(rows):
            times.append(rnd.choice(cols[rows == r]))

        return np.array(times, dtype=int)

    raise ValueError(f"Invalid prediction mode {prediction_rule}")


def mask_test_samples(Y, t_pred, time_gap=3, min_n_train=3):

    # Copy values to be predicted
    y_true = np.copy(Y[range(Y.shape[0]), t_pred])

    # Remove observations over prediction window
    for i_row in range(Y.shape[0]):
        Y[i_row, max(0, t_pred[i_row] - time_gap) :] = 0

    # Find rows that still contain observations
    valid_rows = np.count_nonzero(Y, axis=1) >= min_n_train

    # Remove all rows that dont contain observations
    return Y[valid_rows], t_pred[valid_rows], y_true[valid_rows]
