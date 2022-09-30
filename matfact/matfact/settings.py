import pathlib

BASE_PATH = pathlib.Path(__file__).parents[1]
TEST_PATH = BASE_PATH / "tests"
DATASET_PATH = BASE_PATH / "datasets"
RESULT_PATH = BASE_PATH / "results"
FIGURE_PATH = RESULT_PATH / "figures"
