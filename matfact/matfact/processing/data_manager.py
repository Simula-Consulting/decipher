import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from matfact.processing.pipelines import matfact_pipeline
from matfact.settings import settings


def get_matrix_dimensions(processing_pipeline: Pipeline):
    return (
        len(processing_pipeline["row_assigner"].row_map),
        processing_pipeline["age_bin_assigner"].n_bins,
    )


def load_and_process_screening_data(
    screening_data_path: str = settings.processing.raw_screening_data_path, **kwargs
):
    raw_data = pd.read_csv(screening_data_path, **kwargs)
    return matfact_pipeline.fit_transform(raw_data)


def generate_observation_matrix(
    screening_data_path: str = settings.processing.raw_screening_data_path,
):
    """Function to generate a risk observation matrix from a raw screening file."""
    # TODO: Add edge case handling where there are > 1 screening result for a time bin
    data = load_and_process_screening_data(screening_data_path=screening_data_path)
    n_rows, n_cols = get_matrix_dimensions(matfact_pipeline)

    observation_matrix = np.zeros([n_rows, n_cols])
    risks = data["risk"].to_numpy()
    observation_matrix[data.row, data.bin] = risks
    return observation_matrix
