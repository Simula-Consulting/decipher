import pathlib

import numpy as np

BASE_PATH = pathlib.Path(__file__).parents[1]
TEST_PATH = BASE_PATH / "tests"
DATASET_PATH = BASE_PATH / "datasets"
RESULT_PATH = BASE_PATH / "results"
FIGURE_PATH = RESULT_PATH / "figures"

default_number_of_states = 4


# Defer the import to later to avoid circular import
def get_default_weight_function():
    from matfact.experiments import data_weights

    return data_weights


default_weights = range(1, default_number_of_states + 1)


#### Data generation ####
# Default observation values for five states.
default_observation_probabilities = np.array([0.01, 0.03, 0.08, 0.12, 0.04])

# Minimum number of observations to be considered valid
# Used during data generation
minimum_number_of_observations = 3
