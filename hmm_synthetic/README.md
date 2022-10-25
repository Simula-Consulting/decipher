# Hidden Markov model (HMM)
[![CI](https://github.com/Simula-Consulting/decipher/actions/workflows/python-app.yaml/badge.svg)](https://github.com/Simula-Consulting/decipher/actions/workflows/python-app.yaml)

Data generation using HMM

Implemented is only the synthetic data generation.
Prediction of parameters is _not_ part of this package.

The minimal Python version is 3.10. For other dependencies, refer to `pyproject.toml`.
Dependencies are managed with [poetry](https://python-poetry.org/docs/).

## Installation
```bash
# install poetry, if not installed yet
curl -sSL https://install.python-poetry.org | python3 -
# install project
poetry install
```
Development dependendies can be ignored by appending the last command with `--without dev`.

## Usage
The package is intended as a framework for developers.
Examples of usage to come.

## Development

For a development install, please install the development requirements.
Refer to the root of the repo for setting up Git hooks with Mookme.

### Testing

We use `pytest` to run the tests. You can execute all tests using the command

```
python -m pytest
```

### Continuous integration

All tests are run using GitHub actions on _pull requests_.
