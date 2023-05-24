from pathlib import Path

import numpy as np
import pandas as pd
from processing.pipelines import matfact_pipeline
from processing.settings import settings
from sklearn.pipeline import Pipeline


def load_and_process_screening_data(
    screening_data_path: Path | None = None, **kwargs
) -> tuple[pd.DataFrame, Pipeline]:
    """Function to load and process raw screening data. Returns the processed dataframe
    and the corresponding fitted pipeline.

    `kwargs` are passed to `pd.read_csv`"""
    screening_data_path = (
        screening_data_path or settings.processing.raw_screening_data_path
    )
    raw_data = pd.read_csv(screening_data_path, **kwargs)
    pipeline = matfact_pipeline()
    processed_data = pipeline.fit_transform(raw_data)
    return processed_data, pipeline


def get_matrix_dimensions_from_pipeline(fitted_pipeline: Pipeline) -> tuple[int, int]:
    """Function to return the required dimensions for the observation matrix,
    based on a fitted pipeline."""
    return (
        len(fitted_pipeline["row_assigner"].row_map),
        fitted_pipeline["age_bin_assigner"].n_bins,
    )


def generate_observation_matrix(
    screening_data_path: Path | None = None,
) -> np.ndarray:
    """Function to generate a risk observation matrix from a raw screening file."""
    # TODO: Add edge case handling where there are > 1 screening result for a time bin
    screening_data_path = (
        screening_data_path or settings.processing.raw_screening_data_path
    )
    data, pipeline = load_and_process_screening_data(
        screening_data_path=screening_data_path,
        usecols=settings.processing.column_names.get_screening_columns(),
    )
    n_rows, n_cols = get_matrix_dimensions_from_pipeline(pipeline)

    observation_matrix = np.zeros([n_rows, n_cols])
    risks = data["risk"].to_numpy()
    observation_matrix[data.row, data.bin] = risks
    return observation_matrix
