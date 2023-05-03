from pathlib import Path

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from processing.pipelines import HHMM_pipeline, matfact_pipeline
from processing.settings import settings
from processing.transformers import (
    FolkeregInfoAdder,
    RiskAdderHHMM,
    TestIndexAdder,
    ToExam,
)

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


@pytest.fixture
def screening_data_with_dob() -> pd.DataFrame:
    screening_data = pd.read_csv(settings.processing.raw_screening_data_path)
    tfm = FolkeregInfoAdder()
    return tfm.fit_transform(screening_data)


@pytest.fixture
def exam_data() -> pd.DataFrame:
    screening_data = pd.read_csv(settings.processing.raw_screening_data_path)
    tfm = Pipeline([("folkereg_adder", FolkeregInfoAdder()), ("to_exam", ToExam())])
    return tfm.fit_transform(screening_data)


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


def test_folkereg_adder(screening_data: pd.DataFrame) -> None:
    # TODO: Write more transformer tests and move to separate testing file
    birthdate_adder = FolkeregInfoAdder()
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

    # test that the death column is added
    death_adder = FolkeregInfoAdder(death_column=True)
    df = death_adder.fit_transform(screening_data)
    assert "is_dead" in df


def test_toexam(screening_data_with_dob: pd.DataFrame) -> None:
    """Test that the ToExam transformer works as expected"""
    to_exam = ToExam()
    df = to_exam.fit_transform(screening_data_with_dob)
    assert "exam_date" in df


def test_testindex_adder(exam_data: pd.DataFrame) -> None:
    """Test that the test index is added."""
    testindex_adder = TestIndexAdder()
    df = testindex_adder.fit_transform(exam_data)
    assert "test_index" in df


def test_risk_adder(exam_data: pd.DataFrame) -> None:
    """Test that the risk is added."""
    assert isinstance(exam_data, pd.DataFrame)
    risk_adder = RiskAdderHHMM()
    df = risk_adder.fit_transform(exam_data)
    assert "risk" in df


def test_hhmm_pipeline() -> None:
    """Test that the HHMM pipeline works as expected."""
    screening_data = pd.read_csv(settings.processing.raw_screening_data_path)
    tfm = HHMM_pipeline()
    df = tfm.fit_transform(screening_data)
    assert "risk" in df
    assert "test_index" in df
    assert "exam_date" in df
    assert "is_dead" in df
