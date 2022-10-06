import numpy as np


def predict_proba(Y, M, t_pred, theta, number_of_states):
    """Predict probailities of future results in longitudinal data.

    Args:
        Y: A (M x T) longitudinal data matrix. Each row is a longitudinal vector with
            observed data up to times < t_pred.
        M: The data matrix computed from factor matrices derived from X (M = U @ V.T).
        t_pred: Time of predictions for each row in Y.
        theta: A confidence parameter (estimated from data in utils.py)
        number_of_states: the number of possible states.

    Returns:
        A (M x number_of_states) matrix of probability estimates.
    """

    logl = np.ones((Y.shape[0], M.shape[0]))
    for i, y in enumerate(Y):

        omega = y != 0
        logl[i] = np.sum(
            -1.0 * theta * ((y[omega])[None, :] - M[:, omega]) ** 2, axis=1
        )

    proba_z = np.empty((Y.shape[0], number_of_states))
    domain = np.arange(1, number_of_states + 1)
    for i in range(Y.shape[0]):

        proba_z[i] = np.exp(logl[i]) @ np.exp(
            -1.0 * theta * (M[:, t_pred[i], None] - domain) ** 2
        )

    return proba_z / (np.sum(proba_z, axis=1))[:, None]
