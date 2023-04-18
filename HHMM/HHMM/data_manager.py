from pathlib import Path
import numpy as np
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline

from processing.pipelines import HHMM_pipeline
from processing.settings import settings


def read_and_process_data(
    screening_data_path: Path = settings.processing.raw_screening_data_path, **kwargs
) -> tuple[pd.DataFrame, Pipeline]:
    """Function to load and process raw screening data. Returns the processed dataframe
    and the corresponding fitted pipeline.

    `kwargs` are passed to `pd.read_csv`"""
    raw_data = pd.read_csv(screening_data_path, **kwargs)
    pipeline = HHMM_pipeline()
    processed_data = pipeline.fit_transform(raw_data)
    return processed_data, pipeline


def create_HHMM_lists(processed_data: pd.DataFrame):
    """Function to create lists of the form expected by the HHMM package."""
    PIDS = processed_data.PID.unique()
    K = 3  # number of test types
    S = 4  # number of states

    testTypes, observations, ages, treatment_indx, censor_ages, death_states = [
        [] for _ in range(6)
    ]

    for PID in PIDS:
        df = processed_data[processed_data.PID == PID].sort_values("age")
        n_obs = len(df)

        testTypes_i = df["test_index"].tolist()

        risks = df["risk"].astype(int).tolist()
        risks = [risk - 1 for risk in risks]

        observations_i = [np.zeros([K, S], dtype=int) for _ in range(n_obs)]

        for i, (k, s) in enumerate(zip(testTypes_i, risks)):
            observations_i[i][k, s] = 1

        ages_i = df["age"].tolist()

        testTypes.append(testTypes_i)
        observations.append(observations_i)
        ages.append(ages_i)

        censor_ages.append(df["age"].max())
        death_states.append(df["is_dead"].max())

        treatment_indx.append([])

    return testTypes, observations, ages, treatment_indx, censor_ages, death_states


def save_HHMM_lists(lists):
    """Function to save the HHMM lists with the correct names for the HHMM package."""
    file_names = [
        "mcmcPatientTestTypes",
        "mcmcPatientObservations",
        "mcmcPatientAges",
        "mcmcPatientTreatmentIndx",
        "mcmcPatientCensorDates",
        "mcmcPatientDeathStates",
    ]
    data_location = Path("data")
    for fname, data in zip(file_names, lists):
        with open(data_location / fname, "wb") as location:
            pickle.dump(data, location)
