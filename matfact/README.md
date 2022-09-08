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

### Notes on Apple M1 chips
The current Poetry project is set up on a M1 comptuter. 
As a consequence, the poetry requires `tensorflow-macos` instead of `tensorflow`.
For the future, we will change this to be a conditional dependency, based on the system it is installed on.

## Usage
The project consists of two main components: data generation (`datasets/`) and the model itself (`experiments/`).

### Experiments
Experiments are run from `experiments/main.py`.
Experiment runs are tracked using [MLFlow](https://mlflow.org/); execute `mlflow ui` (in the poetry environment) to open the UI for showing experiment runs, by default available at http://127.0.0.1:5000.

There are multiple entry points to the code, depending on what level of control you need.
For most cases, the function `experiment` in `experiments/main.py` is the most convenient.
For more granular use, `model_factory` is a convenience function for returning appropriate model objects alternatively create models directly.