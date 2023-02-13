from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from matfact.settings import settings


class BirthdateAdder(BaseEstimator, TransformerMixin):
    """Adds a birthdate column to the screening data by using the PID mapping to another
    file containing the birth registry.

    The only valid person status is 'B' meaning 'bosatt', other statuses such as 'dead', 'emigrated', etc.
    are not included and will have a None value in the birthdate column.
    """

    dob_map: dict[int, str] = dict()

    def __init__(
        self,
        birthday_file: str = settings.processing.raw_dob_data_path,
    ) -> None:
        self.birthday_file = birthday_file
        self.dob_data = pd.read_csv(birthday_file)
        self.columns = settings.processing.column_names

    def fit(self, X, y=None) -> BirthdateAdder:
        self.dob_map = (
            self.dob_data[self.dob_data[self.columns.dob.name] == "B"]
            .set_index(self.columns.pid)
            .to_dict()[self.columns.dob.date]
        )
        return self

    def transform(self, X) -> pd.DataFrame:
        X = X.copy()
        X[self.columns.dob.date] = X[self.columns.pid].map(self.dob_map)
        return X


class DatetimeConverter(BaseEstimator, TransformerMixin):
    """Converts specified time columns into datetimes."""

    def __init__(self) -> None:
        self.columns = settings.processing.column_names

    def fit(self, X, y=None) -> DatetimeConverter:
        return self

    def transform(self, X) -> pd.DataFrame:
        def datetime_conversion(x: str) -> pd.Timestamp:
            return pd.to_datetime(x, format=settings.processing.dateformat)

        X = X.copy()
        date_columns = self.columns.get_date_columns()
        X[date_columns] = X[date_columns].apply(datetime_conversion)
        return X


class AgeAdder(BaseEstimator, TransformerMixin):
    """The AgeAdder adds an age column to the dataframe based which is the person age at the the time of
    screening. The age column is therefore based on the cytology/histology date and their birthdate."""

    def __init__(self) -> None:
        self.columns = settings.processing.column_names

    def fit(self, X, y=None) -> AgeAdder:
        return self

    def transform(self, X) -> pd.DataFrame:
        X = X.copy()
        X["age"] = ""
        for col in (self.columns.cyt.date, self.columns.hist.date):
            X.loc[X[col].notna(), "age"] = (X[col] - X[self.columns.dob.date]).apply(
                lambda x: x.days
            )
        return X


class RiskAdder(BaseEstimator, TransformerMixin):
    """The risk adder maps the screening result to a specified risk level (1-4) based on a
    mapping defined in the settings file."""

    def __init__(self) -> None:
        self.risk_maps = settings.processing.risk_maps

    def fit(self, X, y=None) -> RiskAdder:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["risk"] = ""

        for screening in self.risk_maps.keys():
            X.loc[X[screening].notna(), "risk"] = X[screening].map(
                self.risk_maps[screening]
            )
        return X


class InvalidRemover(BaseEstimator, TransformerMixin):
    """Removes invalid rows in the data.

    Rows can be invalid if the person has less than 2 screenings in the dataset, or if there are
    NaN values in the risk or age columns, which means that either the screening result was undefined
    or the person does not currently have a valid status.
    """

    def __init__(self, min_n_tests: int = 2) -> None:
        self.min_n_tests = min_n_tests
        self.columns = settings.processing.column_names

    def fit(self, X, y=None) -> InvalidRemover:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.dropna(subset=["age", "risk"])
        person_counts = X[self.columns.pid].value_counts()
        rejected_pids = person_counts[person_counts.values < self.min_n_tests].index
        out = X[~X[self.columns.pid].isin(rejected_pids)]
        return out


class DataSampler(BaseEstimator, TransformerMixin):
    """A sampler that can reduce the dataset to include only a maximum number of females.

    If max_n_females is set to None, this transformer does nothing, but if there is a need to
    reduce the size of the dataset we can use this to specify a lower number of females to be
    included.
    """

    def __init__(self, max_n_females: int = None):
        self.max_n_females = max_n_females
        self.columns = settings.processing.column_names

        self.n_total = None

    def fit(self, X, y=None) -> DataSampler:
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

    def __init__(self, save_path: str = None) -> None:
        self.save_path = save_path
        self.pid = settings.processing.column_names.pid

    def fit(self, X, y=None) -> RowAssigner:
        individuals = sorted(X[self.pid].unique())
        n_females = len(individuals)
        self.row_map = dict(zip(individuals, np.arange(n_females)))

        if self.save_path is not None:
            with open(self.save_path, "wb") as f:
                pickle.dump(self.row_map, f)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        X["row"] = X[self.pid].map(self.row_map)
        return X
