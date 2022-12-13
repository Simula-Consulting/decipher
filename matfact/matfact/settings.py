import pathlib

import numpy as np

BASE_PATH = pathlib.Path(__file__).parents[1]
TEST_PATH = BASE_PATH / "tests"
DATASET_PATH = BASE_PATH / "datasets"
RESULT_PATH = BASE_PATH / "results"
FIGURE_PATH = RESULT_PATH / "figures"

create_path_default = True  # Create artifact directories if non-existent

default_number_of_states = 4


default_weights = range(1, default_number_of_states + 1)


#### Data generation ####
# Default observation values for five states.
default_observation_probabilities = np.array([0.01, 0.03, 0.08, 0.12, 0.04])

# Minimum number of observations to be considered valid
# Used during data generation
minimum_number_of_observations = 3


### Convergence monitor ###
DEFAULT_NUMBER_OF_EPOCHS = 2000
DEFAULT_EPOCHS_PER_VAL = 5
DEFAULT_PATIENCE = 200


### Projected Gradient Descent
# Pretty much arbitrary values and require hyperparameter search on
# real data.
DEFAULT_TAU = 1.0
DEFAULT_GAMMA = 3.0
