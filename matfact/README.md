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
The current Poetry project is set up on a M1 comptuter.
As a consequence, the poetry requires `tensorflow-macos` instead of `tensorflow`.
For the future, we will change this to be a conditional dependency, based on the system it is installed on.

## Usage
The project consists of two main components: data generation (`datasets/`) and the model itself (`experiments/`).
To run the code, the main entry point is `example.py`.

Experiment runs are tracked using [MLFlow](https://mlflow.org/); execute `mlflow ui` (in the poetry environment) to open the UI for showing experiment runs, by default available at http://127.0.0.1:5000.

There are multiple entry points to the code, depending on what level of control you need.
For an example of using the program, see `main` in `example.py`.
For more granular use, `model_factory` in `experiments/main.py` is a convenience function for returning appropriate model objects alternatively create models directly.

For setting dataset and results paths, go to `settings.py`.


## Future work / development
We here compile a list of incomplete tasks and suggestions for future development.

### Improved test coverage
The test coverage is very low.
In particular, there should be more tests concering the models themselves, `cmf.py`, `wcmf.py`, `scmf.py`.

### Refactor the model code
The models now have some confusing behaviour in that they overwrite their input matrices for some models but not for others.
For example, in the case of SCMF the model.X is modified by the solver.
This should be clearly documented, and preferably rewritten such that the input variables are not modified.
At the very least, the behaviour should be consistent accross all the models.

### Overall improved documentation
The docstring coverage is quite good, but should still be improved.

### Track MLFlow runs in sqlite db

### Figure out the tf.function
See if there is some performance to get from using the tf.function cleverly

### Clean up folder and file structure
Now, the strucuter is `<submodule>/main.py`.
This is not common, and maybe a better solution is to import the exportable things in `__init__.py`, to get more clear improt statements, like `from data_generation import Dataset`, instead of the current `from data_generation.main import Dataset`.
