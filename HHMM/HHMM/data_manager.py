
from pathlib import Path
import numpy as np
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
    PIDS = processed_data.PID.unique()
    N_patients = len(PIDS)
    K = 3
    S = 4

    testTypes, observations, ages, treatment_indx, censor_ages, death_states = [[] for _ in range(6)]

    for PID in PIDS:
        df = processed_data[processed_data.PID == PID].sort_values("age")
        n_obs = len(df)

        testTypes_i = df["test_index"].tolist()

        risks = df["risk"].astype(int).tolist()
        risks = [risk - 1 for risk in risks]

        observations_i = [
            np.zeros([K, S], dtype=int) for _ in range(n_obs)
        ]

        for i, (k, s) in enumerate(zip(testTypes_i, risks)):
            observations_i[i][k, s] = 1

        ages_i = df["age"].tolist()

        testTypes.append(testTypes_i)
        observations.append(observations_i)
        ages.append(ages_i)

        censor_ages.append(df["age"].max())
        death_states.append(df["is_dead"].max())

        treatment_indx.append([])

    return PIDS.tolist(), testTypes, observations, ages, treatment_indx, censor_ages, death_states
