import argparse
import pathlib

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("MACOSX")
import logging

from matfact.experiments.algorithms.risk_prediction import predict_proba, predict_state
from matfact.experiments.simulation.dataset import _t_pred


def plot_traj(states, ignore_all_normal=True):
    matplotlib.use("MACOSX")
    for i, state in enumerate(states.T):
        if not np.all(state == 1):
            plt.scatter(np.arange(len(state)), i * np.ones_like(state), c=state)
    plt.show()


def plot_traj(states, ignore_all_normal=True):
    matplotlib.use("MACOSX")
    mask = np.any(states != 1, axis=1)
    plt.imshow(states[np.nonzero(mask)].T, aspect="auto", interpolation=None)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="MatFact module.")
    parser.add_argument(
        "model_array", type=pathlib.Path, help="path to the model array."
    )
    parser.add_argument(
        "observation_array", type=pathlib.Path, help="path to the observation array."
    )
    parser.add_argument("--theta", type=float, default=5)
    parser.add_argument("--num", type=int, default=1)
    parser.add_argument("--step", type=int, default=1)

    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    try:
        model_array = np.load(args.model_array)
        observation_array = np.load(args.observation_array)
    except ValueError as e:
        logging.exception("The supplied files could not be read as arrays!")
        exit(1)

    if not model_array.shape == observation_array.shape:
        raise ValueError(
            "The supplied matrices have different shapes! "
            "{model_array.shape} vs. {observation_array.shape}"
        )

    number_of_individuals, time_steps = model_array.shape

    predicted_states = np.empty((args.num, number_of_individuals))
    time_to_predict = _t_pred(observation_array, prediction_rule="last_observed")
    if np.max(time_to_predict + args.num * args.step > time_steps):
        raise ValueError(
            "The given parameters would imply predicting outside of the training interval."
        )

    for i in range(args.num):
        probabilities = predict_proba(
            observation_array, model_array, time_to_predict + i * args.step, args.theta
        )
        predicted_states[i] = predict_state(probabilities, None)

    print(np.diff(predicted_states, axis=0))
    plot_traj(predicted_states)


if __name__ == "__main__":
    main()
