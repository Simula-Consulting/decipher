# HHMM Usage
This document describes how to use the HHMM package for simple data processing, running the [HHMM inference algorithm](https://github.com/Corleno/HHMM) from R. Meng et al. (2022).

## Install dependencies
To install the project dependencies, run the following command in the root folder of the project:
```bash
poetry install
```
then activate the environment with

```bash
poetry shell
```

## Set up data processing
In the processing settings `../processing/processing/settings.py`, ensure that the `raw_screening_data_path` and `raw_dob_data_path` points to the correct file locations.

For testing purposes, you can use the mocked test datasets:
```python
file_location: Path = Path("../processing/tests/test_datasets")
raw_screening_data_path: Path = file_location / "test_screening_data.csv"
raw_dob_data_path: Path = file_location / "test_dob_data.csv"
```

Then prepare the data in using the functions in `data_manager.py` as follows:
```python
from HHMM.data_manager import (
    read_and_process_data,
    create_HHMM_lists,
    save_HHMM_lists
)

# Read and process data
processed_data, pipeline = read_and_process_data()

# Create HHMM lists
HHMM_lists = create_HHMM_lists(processed_data)

# Save HHMM lists
save_HHMM_lists(HHMM_lists)
```
This will create 6 pickle files in the `../data/` folder with the appropriate names for the HHMM inference algorithm.

## Run HHMM inference algorithm
Run the HHMM inference algorithm:
```bash
python HHMM/inference.py --max_steps_EM <max_steps_EM>
```
where `<max_steps_EM>` is the maximum number of EM steps to run. The default is 100, but this takes a long time. For testing purposes, you can use 1 or 2.

This will save the results in the `outcome/` folder.

## Run HHMM prediction algorithm
Run the HHMM prediction algorithm:
```bash
python HHMM/prediction.py
```
which will save two files in the `res/` folder:
- `hierarchical_prediction_LS.pickle` which is the predicted final state of each patient.
- `hierarchical_prediction_ML.pickle` which is the predicted posterior for the frailty class of each patient.
