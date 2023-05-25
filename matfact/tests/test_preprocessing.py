from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from processing.pipelines import matfact_pipeline
from processing.settings import settings
from sklearn.pipeline import Pipeline

from matfact.preprocessing.data_manager import generate_observation_matrix

settings.processing.raw_dob_data_path = Path(
    "../processing/tests/test_datasets/test_dob_data.csv"
)
settings.processing.raw_screening_data_path = Path(
    "../processing/tests/test_datasets/test_screening_data.csv"
)


@pytest.fixture
def testing_pipeline() -> Pipeline:
    return matfact_pipeline(min_n_tests=0, verbose=False)


@pytest.fixture
def screening_data() -> pd.DataFrame:
    return pd.read_csv(settings.processing.raw_screening_data_path)


def test_generate_observation_matrix(
    screening_data: pd.DataFrame, testing_pipeline: Pipeline
) -> None:
    reference = testing_pipeline.fit_transform(screening_data)
    X = generate_observation_matrix()
    n_rows, n_cols = X.shape

    assert n_rows == reference[settings.processing.column_names.pid].nunique()
    assert n_cols == testing_pipeline["age_bin_assigner"].n_bins
    assert np.all((X >= 0) & (X <= 4))
