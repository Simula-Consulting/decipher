# MATFACT -- MATrix FACtorization for cervical cancer screening data reconstruction and prediction

## Installation
This project is managed using [Poetry](https://python-poetry.org/).
To set it up, you must have Poetry installed.
Then, navigate to the project root directory, and run `poetry install`.

```python
# To run programs, either do so through Poetry
poetry run python <filename.py>

# or enter the Poetry shell
poetry shell
python <filename.py>
```

### Notes on Apple Silicon chips (M1, M2, etc.)
The current Poetry project is set up on an M1 computer.
As a consequence, the poetry requires `tensorflow-macos` instead of `tensorflow`.
For the future, we will change this to be a conditional dependency, based on the system it is installed on.

## Usage
The project source code is located in under `matfact`.
The code consists of three main modules: `data_generation`, `experiments`, and `plotting`.
The factorisation and prediction, i.e. train and test, is part of `experiments`.
As a starting point on how to use the library, example scripts are found in `examples`.
`example.py` and `example_train_and_log.py` shows data generation, factorization, prediction, and plotting, with the latter utilizing the convenience funtion `train_and_log`, which is reccomended.
`example_hyperparamsearch.py` shows how one may perform a simple hyperparameter search.

Experiment runs are tracked using [MLFlow](https://mlflow.org/); execute `mlflow ui` (in the poetry environment) to open the UI for showing experiment runs, by default available at http://127.0.0.1:5000.

For setting dataset and results paths, go to `settings.py`.
