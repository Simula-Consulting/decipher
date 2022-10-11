import numpy as np

from matfact.experiments.predict.clf_tree import ClassificationTree


def predict_proba(Y, M, t_pred, theta, domain=np.arange(1, 5)):
    """Predict probailities of future results in longitudinal data.

    Args:
        Y: A (M x T) longitudinal data matrix. Each row is a longitudinal vector with
            observed data up to times < t_pred.
        M: The data matrix computed from factor matrices derived from X (M = U @ V.T).
        t_pred: Time of predictions for each row in Y.
        theta: A confidence parameter (estimated from data in utils.py)
        domain: The possible values at t_pred

    Returns:
        A (M x domain.size) matrix of probability estimates.
    """

    logl = np.ones((Y.shape[0], M.shape[0]))
    for i, y in enumerate(Y):

        omega = y != 0
        logl[i] = np.sum(
            -1.0 * theta * ((y[omega])[None, :] - M[:, omega]) ** 2, axis=1
        )

    proba_z = np.empty((Y.shape[0], domain.shape[0]))
    for i in range(Y.shape[0]):

        proba_z[i] = np.exp(logl[i]) @ np.exp(
            -1.0 * theta * (M[:, t_pred[i], None] - domain) ** 2
        )

    return proba_z / (np.sum(proba_z, axis=1))[:, None]


def predict_state(
    probability_predictions: np.ndarray, estimator: ClassificationTree | None = None
):
    """Predict the state given probability predictions for the different states.

    Arguments:
     probability_predictions: (number_of_samples x number_of_states) ndarray
     estimator: sklearn API compliant estimator. If None, the simple estimator
     of chosing the state with the highest probability is used."""

    if estimator:
        return estimator.predict(probability_predictions)
    else:
        return np.argmax(probability_predictions, axis=1) + 1  # States are 1-indexed
