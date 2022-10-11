import argparse
import pathlib

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml

from matfact.experiments.predict.clf_tree import ClassificationTree

matplotlib.use("MACOSX")
import logging

from matfact.experiments.algorithms.risk_prediction import (
    fill_history,
    predict_proba,
    predict_state,
)
from matfact.experiments.simulation.dataset import _t_pred


class ConfigurationError(Exception):
    """The configuration is not valid."""

    pass


def plot_traj(states, ignore_all_normal=True):
    matplotlib.use("MACOSX")
    for i, state in enumerate(states.T):
        if not np.all(state == 1):
            plt.scatter(np.arange(len(state)), i * np.ones_like(state), c=state)
    plt.show()


def plot_traj(states, ignore_all_normal=True):
    matplotlib.use("MACOSX")
    mask = np.nonzero(np.any(states != 1, axis=1))
    plt.imshow(states[mask], aspect="auto", interpolation=None)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="MatFact module.")
    parser.add_argument(
        "model_array", type=pathlib.Path, help="path to the model array."
    )
    parser.add_argument(
        "observation_array", type=pathlib.Path, help="path to the observation array."
    )
    parser.add_argument("--config", type=pathlib.Path, help="config file")
    parser.add_argument("--theta", type=float, default=5)
    parser.add_argument("--num", type=int, default=1)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--plot", action="store_true")

    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()

    if not all(
        file is None or file.is_file()
        for file in [args.model_array, args.observation_array, args.config]
    ):
        raise ValueError("Supply files, not directories!")

    predicted_states_path = args.observation_array.with_stem(
        args.observation_array.stem + "_predicted"
    )
    if args.plot:
        predicted_states = np.load(predicted_states_path)
        plot_traj(predicted_states)
        exit()

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

    config = {}
    if args.config:
        with open(args.config) as config_file:
            config.update(yaml.full_load(config_file))

    if config.get("thresholds") and config.get("number_of_states"):
        if len(config["thresholds"]) != config["number_of_states"]:
            raise ConfigurationError(
                "The number of threshold values does not match the number of states."
            )

    estimator = (
        ClassificationTree(thresholds)
        if (thresholds := config.get("thresholds"))
        else None
    )

    predicted_states = fill_history(
        observation_array,
        model_array,
        args.theta,
        estimator=estimator,
        use_predictions_as_observations=config.get(
            "use_prediction_as_observations", False
        ),
    )
    np.save(predicted_states_path, predicted_states)
    # plot_traj(predicted_states)


if __name__ == "__main__":
    main()
