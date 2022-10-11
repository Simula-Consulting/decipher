import numpy as np

from matfact.experiments.predict.clf_tree import ClassificationTree
from matfact.experiments.simulation.dataset import _t_pred


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
     of choosing the state with the highest probability is used."""

    if estimator:
        return estimator.predict(probability_predictions)
    else:
        return np.argmax(probability_predictions, axis=1) + 1  # States are 1-indexed


def fill_history(
    observation_array: np.ndarray,
    model_array: np.ndarray,
    theta: float,
    use_predictions_as_observations: bool = False,
    estimator: ClassificationTree | None = None,
):
    """Given an observation history, fill the states after the last observation.

    Arguments:
     observation_array: (number_of_individuals x time_slots) ndarray of observations
     model_array: (number_of_individuals x time_slots) ndarray from training
     theta: theta value of prediction algorithm
     use_predictions_as_observations: if True, the predicted states will be used for
        the consecutive predictions.
     estimator: the estimator to be used by predict_states.

    Returns:
     (number_of_individuals x time_slots) ndarray with the predicted states. NB! the
        array only contains the predicted values, the actual observations are not
        included."""
    time_steps = observation_array.shape[1]
    predicted_states = np.zeros_like(observation_array)

    # The first time slot after the last observation
    time_after_last = _t_pred(observation_array, prediction_rule="last_observed") + 1
    for i in range(time_steps - np.min(time_after_last)):
        time_to_predict = time_after_last + i
        mask = np.nonzero(time_to_predict < time_steps)
        observed = (
            observation_array[mask] + predicted_states[mask]
            if use_predictions_as_observations
            else observation_array[mask]
        )
        probabilities = predict_proba(
            observed, model_array[mask], time_after_last[mask] + i, theta
        )
        predicted_states[mask, time_to_predict[mask]] = predict_state(
            probabilities, estimator=estimator
        )
    return predicted_states
