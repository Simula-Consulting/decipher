from pathlib import Path

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from processing.pipelines import matfact_pipeline
from processing.settings import settings
from processing.transformers import BirthdateAdder

settings.processing.raw_dob_data_path = Path("tests/test_datasets/test_dob_data.csv")
settings.processing.raw_screening_data_path = Path(
    "tests/test_datasets/test_screening_data.csv"
)


@pytest.fixture
def testing_pipeline() -> Pipeline:
    return matfact_pipeline(min_n_tests=0, verbose=False)


@pytest.fixture
def screening_data() -> pd.DataFrame:
    return pd.read_csv(settings.processing.raw_screening_data_path)


def test_processing_pipeline(
    screening_data: pd.DataFrame, testing_pipeline: Pipeline
) -> None:
    prepared_data = testing_pipeline.fit_transform(screening_data)
    added_cols = [
        settings.processing.column_names.dob.date,
        "age",
        "risk",
        "bin",
        "row",
    ]
    assert prepared_data[added_cols].isna().sum().sum() == 0
    assert all(prepared_data[added_cols[1:]] > 0)

    for _, data in prepared_data.groupby(settings.processing.column_names.pid):
        assert len(data["row"].unique()) == 1


def test_birthdate_adder(screening_data: pd.DataFrame) -> None:
    # TODO: Write more transformer tests and move to separate testing file
    birthdate_adder = BirthdateAdder()
    df = birthdate_adder.fit_transform(screening_data)
    column_names = settings.processing.column_names
    date_col_name = column_names.dob.date

    assert date_col_name not in screening_data
    assert date_col_name in df

    assert (
        df[df[column_names.pid].isin(birthdate_adder.dob_map.keys())][date_col_name]
        .isna()
        .sum()
        == 0
    )
