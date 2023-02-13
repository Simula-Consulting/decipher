import pickle

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from matfact.settings import settings


class BirthdateAdder(BaseEstimator, TransformerMixin):
    def __init__(self, birthday_file: str = settings.raw_dob_data_path) -> None:
        self.birthday_file = birthday_file
        self.dob_data = pd.read_csv(self.birthday_file)
        self.dob_map: dict[int, str] = None
        self.columns = settings.processing.column_names

    def fit(self, X, y=None):
        self.dob_map = (
            self.dob_data[self.dob_data[self.columns.dob.name] == "B"]
            .set_index(self.columns.pid)
            .to_dict()[self.columns.dob.date]
        )
        return self

    def transform(self, X):
        X = X.copy()
        X[self.columns.dob.date] = X[self.columns.pid].map(self.dob_map)
        return X


class DatetimeConverter(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns = settings.processing.column_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def datetime_conversion(x: str) -> pd.Timestamp:
            return pd.to_datetime(x, format=settings.processing.dateformat)

        X = X.copy()
        date_columns = self.columns.get_date_columns()
        X[date_columns] = X[date_columns].apply(datetime_conversion)
        return X


class AgeAdder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns = settings.processing.column_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["age"] = ""
        for col in (self.columns.cyt.date, self.columns.hist.date):
            X.loc[X[col].notna(), "age"] = (X[col] - X[self.columns.dob.date]).apply(
                lambda x: x.days
            )
        return X


class RiskAdder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.risk_maps = settings.processing.risk_maps

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        X["risk"] = ""

        for screening in self.risk_maps.keys():
            X.loc[X[screening].notna(), "risk"] = X[screening].map(
                self.risk_maps[screening]
            )
        return X


class InvalidRemover(BaseEstimator, TransformerMixin):
    def __init__(self, min_n_tests: int = 2):
        self.min_n_tests = min_n_tests
        self.columns = settings.processing.column_names

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X = X.dropna(subset=["age", "risk"])
        person_counts = X[self.columns.pid].value_counts()
        rejected_pids = person_counts[person_counts.values < self.min_n_tests].index
        return X[~X[self.columns.pid].isin(rejected_pids)]


class SampleFemales(BaseEstimator, TransformerMixin):
    def __init__(self, max_n_females: int = None):
        self.max_n_females = max_n_females
        self.columns = settings.processing.column_names

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        if self.max_n_females is None:
            return X

        X = X.copy()
        n_total = X[self.columns.pid].nunique()
        n_females = self.max_n_females if self.max_n_females <= n_total else n_total

        selected = np.random.choice(
            X[self.columns.pid].unique(),
            size=n_females,
            replace=False,
        )
        selected.sort()
        X = X[X[self.columns.pid].isin(selected)]
        return X


class AgeBinAssigner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.n_bins: int = None

    def fit(self, X: pd.DataFrame, y=None):
        age_min, age_max = X["age"].min(), X["age"].max()
        avg_days_per_month = 30.437

        def ceildiv(a: float, b: float) -> int:
            """Function to perform ceiling division, opposite of floor division."""
            return int(-(a // -b))

        self.n_bins = ceildiv(
            age_max - age_min,
            settings.processing.months_per_timepoint * avg_days_per_month,
        )
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        _, bin_edges = np.histogram(X["age"], bins=self.n_bins)
        bin_edges[0] -= 1  # to make sure the youngest age is included
        indexes = np.arange(self.n_bins)
        X["bin"] = pd.cut(X["age"], bins=bin_edges, labels=indexes)
        return X


class RowAssigner(BaseEstimator, TransformerMixin):
    def __init__(self, save_path: str = None):
        self.save_path = save_path
        self.pid = settings.processing.column_names.pid

        self.row_map: dict[int, int] = None

    def fit(self, X, y=None):
        individuals = sorted(X[self.pid].unique())
        n_females = len(individuals)
        self.row_map = dict(zip(individuals, np.arange(n_females)))

        if self.save_path is not None:
            with open(self.save_path, "wb") as f:
                pickle.dump(self.row_map, f)
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()

        X["row"] = X[self.pid].map(self.row_map)
        return X
