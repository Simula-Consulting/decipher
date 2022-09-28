# hmm_synthetic
[![CI](https://github.com/Simula-Consulting/hmm_synthetic/actions/workflows/test.yml/badge.svg)](https://github.com/Simula-Consulting/hmm_synthetic/actions/workflows/test.yml)

Data generation using HMM

The minimal Python version is 3.8. For other dependencies, refer to `pyproject.toml`.
Dependencies are managed with [poetry](https://python-poetry.org/docs/).

## Installation
```bash
# install poetry, if not installed yet
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
# install project
poetry install
```
Development dependendies can be ignored by appending the last command with `--no-dev`.

## Usage
After installing, run
```
python main.py
```

## Development

For a development install, please install the development requirements.
Then install the pre-commit hook (<https://pre-commit.com/>):
```
pre-commit install
```

### Testing

We use `pytest` to run the tests. You can execute all tests using the command

```
python -m pytest
```

### Continuous integration

All tests are run using GitHub actions on _push_ and _pull requests_. 
