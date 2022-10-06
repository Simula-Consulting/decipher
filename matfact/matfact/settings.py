import pathlib

import numpy as np

BASE_PATH = pathlib.Path(__file__).parents[1]
TEST_PATH = BASE_PATH / "tests"
DATASET_PATH = BASE_PATH / "datasets"
RESULT_PATH = BASE_PATH / "results"
FIGURE_PATH = RESULT_PATH / "figures"

#### Data generation ####
# Default observation values for five states.
default_observation_probabilities = np.array([0.01, 0.03, 0.08, 0.12, 0.04])
