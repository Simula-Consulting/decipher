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
To run the code, the main entry point is `main.py`.

Experiment runs are tracked using [MLFlow](https://mlflow.org/); execute `mlflow ui` (in the poetry environment) to open the UI for showing experiment runs, by default available at http://127.0.0.1:5000.

There are multiple entry points to the code, depending on what level of control you need.
For most cases, the function `main` in `main.py` is the most convenient, alternatively `experiment` in the same file offers some more control.
For more granular use, `model_factory` in `experiments/main.py` is a convenience function for returning appropriate model objects alternatively create models directly.