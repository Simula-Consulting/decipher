import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from matfact.processing.transformers import (
    AgeAdder,
    AgeBinAssigner,
    BirthdateAdder,
    DataSampler,
    DatetimeConverter,
    InvalidRemover,
    RiskAdder,
    RowAssigner,
)
from matfact.settings import settings

# Changing default data paths to testing datasets
settings.raw_screening_data_path = "tests/test_datasets/test_screening_data.csv"
settings.raw_dob_data_path = "tests/test_datasets/test_dob_data.csv"


def create_custom_pipeline(*, n_females: int | None = None, min_n_tests: int = 0):
    """Function to return the matfact processing pipeline but with custom arguments for testing."""
    return Pipeline(
        [
            ("birthdate_adder", BirthdateAdder()),
            ("datetime_converter", DatetimeConverter()),
            ("age_adder", AgeAdder()),
            ("risk_adder", RiskAdder()),
            ("invalid_remover", InvalidRemover(min_n_tests=min_n_tests)),
            (
                "data_sampler",
                DataSampler(max_n_females=n_females),
            ),
            ("age_bin_assigner", AgeBinAssigner()),
            (
                "row_assigner",
                RowAssigner(),
            ),
        ]
    )


@pytest.fixture
def screening_data() -> pd.DataFrame:
    return pd.read_csv(settings.raw_screening_data_path)


def test_processing_pipeline(screening_data: pd.DataFrame) -> None:
    processing_pipeline = create_custom_pipeline()
    prepared_data = processing_pipeline.fit_transform(screening_data)
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
