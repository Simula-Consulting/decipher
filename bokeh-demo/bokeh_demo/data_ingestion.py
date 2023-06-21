from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


class NAHandler(BaseEstimator, TransformerMixin):
    """Transformer to transform pandas boolean columns into standard columns with
    numpy nans instead of pd._libs.missing.NAType which is not supported by the app."""

    def __init__(self, bool_cols: list[str] | None = None) -> None:
        self.bool_cols = bool_cols or []

    def fit(self, X: pd.DataFrame, y=None) -> NAHandler:
        self.bool_cols += X.select_dtypes(include=["bool"]).columns.to_list()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.bool_cols:
            X[self.bool_cols] = X[self.bool_cols].astype("float")
        return X


class CategoryColumnConverter(BaseEstimator, TransformerMixin):
    """Transforms category columns into string columns due to Bokeh not accepting category columns.
    If the column values are enums, they are converted to their values.
    """

    category_cols: list[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> CategoryColumnConverter:
        self.category_cols = X.select_dtypes(include=["category"]).columns
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.category_cols:
            X[col] = X[col].apply(self.try_convert_enum)
        X[self.category_cols] = X[self.category_cols].astype("str")
        return X

    @staticmethod
    def try_convert_enum(x):
        """Converts enum to its value if it has one, otherwise returns the value itself."""
        try:
            return x.value
        except AttributeError:
            return x


class CreatePersonSource(BaseEstimator, TransformerMixin):
    """Takes in the exams dataframe and returns another dataframe (with one row per person)
    for plotting Lexis plots."""

    # Create a map from person id to exam results, ages at exams, and the indices of the exams
    person_results_map: dict[int, list[int]]
    person_exam_ages_map: dict[int, list[int]]
    person_exam_inds_map: dict[int, list[int]]

    def fit(self, X, y=None) -> CreatePersonSource:
        self.person_results_map = {
            pid: sub_df["risk"].to_list() for pid, sub_df in X.groupby("PID")
        }
        self.person_exam_ages_map = {
            pid: (sub_df["age"] / pd.Timedelta(days=365)).to_list()
            for pid, sub_df in X.groupby("PID")
        }
        self.person_exam_inds_map = {
            pid: (sub_df.index.to_list()) for pid, sub_df in X.groupby("PID")
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Generate the person dataframe based on the exams dataframe
        person_df = (
            X.groupby("PID")
            .agg(
                {
                    "exam_date": self._minmax,
                    "age": self._minmax,
                    "FOEDT": "first",  # We assume all FOEDT are the same
                }
            )
            .reset_index()  # We want PID as explicit column
        )

        person_df = person_df.rename(
            columns={
                "exam_date": "lexis_line_endpoints_year",
                "age": "lexis_line_endpoints_age",
            }
        )

        person_df["lexis_line_endpoints_person_index"] = person_df["PID"].transform(
            lambda pid: (pid, pid)
        )

        # Adding per person exam result and ages at exam time
        person_df["exam_results"] = person_df["PID"].map(self.person_results_map)
        person_df["exam_time_age"] = person_df["PID"].map(self.person_exam_ages_map)
        person_df["exam_idx"] = person_df["PID"].map(self.person_exam_inds_map)

        # Adding dummy vaccine data for future use
        person_df[["vaccine_age", "vaccine_year", "vaccine_type"]] = None
        person_df[["vaccine_line_endpoints_age", "vaccine_line_endpoints_year"]] = [
            ([], []) for _ in range(person_df.shape[0])
        ]
        return person_df

    @staticmethod
    def _minmax(column):
        return (min(column), max(column))


def _sort_by_exam_date(X: pd.DataFrame) -> pd.DataFrame:
    return X.sort_values(by="exam_date").reset_index(drop=True)


exams_pipeline = Pipeline(
    [
        ("date_sorter", FunctionTransformer(_sort_by_exam_date)),
        ("NA_handler", NAHandler(bool_cols=["risk"])),
        ("category_converter", CategoryColumnConverter()),
    ]
)
