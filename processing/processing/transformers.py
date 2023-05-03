from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from processing.settings import settings


class FolkeregInfoAdder(BaseEstimator, TransformerMixin):
    """Adds a birthdate column and a death/emigrated indicator to the screening data by using the PID mapping to another
    file containing the birth registry."""

    dob_map: dict[int, str] = dict()
    death_map: dict[int, int] = dict()

    def __init__(
        self,
        birthday_file: Path | None = None,
        death_column: bool = True,
    ) -> None:
        self.birthday_file = birthday_file or settings.processing.raw_dob_data_path
        self.dob_data = pd.read_csv(self.birthday_file)
        self.columns = settings.processing.column_names

        self.death_column = death_column

    def fit(self, X: pd.DataFrame, y=None) -> FolkeregInfoAdder:
        self.dob_map = self.dob_data.set_index(self.columns.pid).to_dict()[
            self.columns.dob.date
        ]
        if self.death_column:
            self.dob_data["is_dead"] = (
                self.dob_data[self.columns.dob_status.date].notna().astype(int)
            )
            self.death_map = self.dob_data.set_index(self.columns.pid).to_dict()[
                "is_dead"
            ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.columns.dob.date] = X[self.columns.pid].map(self.dob_map)

        if self.death_column:
            X["is_dead"] = X[self.columns.pid].map(self.death_map)
        return X


class DatetimeConverter(BaseEstimator, TransformerMixin):
    """Converts specified time columns into datetimes."""

    def __init__(self, columns: list[str] = None) -> None:
        self.columns = settings.processing.column_names.get_date_columns()
        if columns is not None:
            self.columns.extend(columns)

    def fit(self, X: pd.DataFrame, y=None) -> DatetimeConverter:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        def datetime_conversion(x: str) -> pd.Timestamp:
            return pd.to_datetime(x, format=settings.processing.dateformat)

        X = X.copy()
        date_columns = list(set(self.columns) & set(X.columns))
        X[date_columns] = X[date_columns].apply(datetime_conversion)
        return X


class AgeAdder(BaseEstimator, TransformerMixin):
    """The AgeAdder adds an age column to the dataframe based which is the person age at the the time of
    screening / exam result.

    The age column is based on a reference column / usually birthdate, and a target column of dates.
    """

    def __init__(
        self,
        *,
        in_years: bool = False,
        target_columns: list[str] | None = None,
        reference_column: str = None,
    ) -> None:
        self.columns = settings.processing.column_names
        self.in_years = in_years
        self.target_columns = target_columns or [
            self.columns.cyt.date,
            self.columns.hist.date,
        ]
        self.reference_column = reference_column or self.columns.dob.date

    def fit(self, X: pd.DataFrame, y=None) -> AgeAdder:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["age"] = np.nan
        div_factor = 365.0 if self.in_years else 1.0
        for col in self.target_columns:
            X.loc[X[col].notna(), "age"] = (X[col] - X[self.reference_column]).apply(
                lambda x: x.days / div_factor
            )
        return X


class RiskAdder(BaseEstimator, TransformerMixin):
    """The risk adder maps the screening result to a specified risk level (1-4) based on a
    mapping defined in the settings file."""

    def __init__(self) -> None:
        self.risk_maps = settings.processing.risk_maps

    def fit(self, X: pd.DataFrame, y=None) -> RiskAdder:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["risk"] = np.nan

        for screening_type in self.risk_maps.keys():
            X.loc[X[screening_type].notna(), "risk"] = X[screening_type].map(
                self.risk_maps[screening_type]
            )
        return X


class RiskAdderHHMM(BaseEstimator, TransformerMixin):
    """Adds a risk column with an integer (1-4) risk level for each exam diagnosis result."""

    def __init__(self) -> None:
        self.risk_map = {
            k: v
            for subdict in settings.processing.risk_maps.values()
            for k, v in subdict.items()
        }

    def fit(self, X: pd.DataFrame, y=None) -> RiskAdderHHMM:
        if "exam_diagnosis" not in X:
            raise ValueError(
                "'exam_diagnosis' column not found. Make sure the DataFrame is transformed to exam-wise."
            )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["risk"] = X["exam_diagnosis"].map(self.risk_map)
        return X


class InvalidRemover(BaseEstimator, TransformerMixin):
    """Removes invalid rows in the data.

    Rows can be invalid if the person has less than 2 screenings in the dataset, or if there are
    NaN values in the risk or age columns, which means that either the screening result was undefined
    or the person does not currently have a valid status.
    """

    def __init__(self, min_n_tests: int | None = None) -> None:
        self.min_n_tests = min_n_tests or settings.processing.min_n_tests
        self.columns = settings.processing.column_names

    def fit(self, X, y=None) -> InvalidRemover:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = X.dropna(subset=["age", "risk"])
        person_counts = X[self.columns.pid].value_counts()
        rejected_pids = person_counts[person_counts.values < self.min_n_tests].index
        X = X[~X[self.columns.pid].isin(rejected_pids)]
        return X


class DataSampler(BaseEstimator, TransformerMixin):
    """A sampler that can reduce the dataset to include only a maximum number of females.

    If max_n_females is set to None, this transformer does nothing, but if there is a need to
    reduce the size of the dataset we can use this to specify a lower number of females to be
    included.
    """

    def __init__(self, max_n_females: int | None = None) -> None:
        self.max_n_females = max_n_females or settings.processing.max_n_females
        self.columns = settings.processing.column_names

        self.n_total = None

    def fit(self, X: pd.DataFrame, y=None) -> DataSampler:
        if self.max_n_females is None:
            return self
        n_total = X[self.columns.pid].nunique()
        self.max_n_females = (
            self.max_n_females if self.max_n_females <= n_total else n_total
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.max_n_females is None:
            return X

        X = X.copy()
        selected = np.random.choice(
            X[self.columns.pid].unique(),
            size=self.max_n_females,
            replace=False,
        )
        selected.sort()
        X = X[X[self.columns.pid].isin(selected)]
        return X


class AgeBinAssigner(BaseEstimator, TransformerMixin):
    """The age assigns an age bin (column) for the each screening.

    The appropriate number of columns required to span the age range of the dataset is calculated
    based on the required resolution (months per timepoint). Then a 'bin' column is added to the dataframe
    which is the column position of each screening.
    """

    n_bins: int = 0

    def fit(self, X: pd.DataFrame, y=None) -> AgeBinAssigner:
        age_min, age_max = X["age"].min(), X["age"].max()
        avg_days_per_month = 30.437

        def ceildiv(a: float, b: float) -> int:
            """Function to perform ceiling division, opposite of floor division."""
            return int(-(a // -b))

        self.n_bins = max(
            ceildiv(
                age_max - age_min,
                settings.processing.months_per_timepoint * avg_days_per_month,
            ),
            1,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        _, bin_edges = np.histogram(X["age"], bins=self.n_bins)
        bin_edges[0] -= 1  # to make sure the youngest age is included
        indexes = np.arange(self.n_bins)
        X["bin"] = pd.cut(X["age"], bins=bin_edges, labels=indexes)
        return X


class RowAssigner(BaseEstimator, TransformerMixin):
    """Assigns a unique row for each female in the dataset and adds this as a 'row' column to the dataframe.

    Has the option to store the row map which would be necessary information to keep in a practical application.
    """

    row_map: dict[int, int] = dict()

    def __init__(self, row_map_save_path: Path | None = None) -> None:
        self.row_map_save_path = (
            row_map_save_path or settings.processing.row_map_save_location
        )
        self.pid = settings.processing.column_names.pid

    def fit(self, X: pd.DataFrame, y=None) -> RowAssigner:
        individuals = sorted(X[self.pid].unique())
        n_females = len(individuals)
        self.row_map = dict(zip(individuals, np.arange(n_females)))

        if self.row_map_save_path is not None:
            with open(self.row_map_save_path, "wb") as f:
                pickle.dump(self.row_map, f)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["row"] = X[self.pid].map(self.row_map)
        return X


class ToExam(BaseEstimator, TransformerMixin):
    """Transform the screening data from screening-based to exam result-based.
    The resulting Dataframe will have one row per exam result.
    """

    def __init__(self, fields_to_keep: list[str] | None = None) -> None:
        self.fields_to_keep = fields_to_keep or ["PID", "FOEDT"]

    def fit(self, X: pd.DataFrame, y=None) -> ToExam:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        mapper = {
            "cytDate": "cytMorfologi",
            "histDate": "histMorfologi",
            "hpvDate": "hpvResultat",
        }

        # Transform from wide to long
        exams = (
            X.reset_index()
            .melt(
                id_vars="index",
                value_vars=mapper.keys(),  # type: ignore[arg-type]
                var_name="exam_type",
                value_name="exam_date",
            )
            .dropna()
            .astype({"exam_type": "category"})
        )

        # Join on result columns
        exams = exams.join(X[mapper.values()], on="index")  # type: ignore[call-overload]

        # Add result column
        conditions = [exams["exam_type"] == key for key in mapper]
        values = [exams[key] for key in mapper.values()]
        exams["exam_diagnosis"] = np.select(conditions, values)

        # Drop the raw exam result
        exams = exams.drop(columns=mapper.values())

        # Remap exam types
        exams["exam_type"] = exams["exam_type"].transform(self._map_exam_type)

        return exams.join(X[self.fields_to_keep], on="index")

    @staticmethod
    def _map_exam_type(field_name):
        return {
            "cytDate": "cytology",
            "histDate": "histology",
            "hpvDate": "hpv",
        }[field_name]


class TestIndexAdder(BaseEstimator, TransformerMixin):
    """Adds a test index to a DataFrame, needed for HHMM code."""

    __test__ = False

    def __init__(self) -> None:
        self.test_index = {"cytology": 0, "histology": 1, "hpv": 2}

    def fit(self, X: pd.DataFrame, y=None) -> TestIndexAdder:
        if "exam_type" not in X:
            raise ValueError(
                "'exam_type' column not found. Make sure the DataFrame is transformed to exam-wise."
            )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["test_index"] = X["exam_type"].map(self.test_index)
        return X
