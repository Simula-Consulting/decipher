import numpy as np
import pandas as pd

from matfact.settings import settings


class ScreeningDataProcessingPipeline:
    """Class to process raw screening data into an observation matrix of observed risk states"""

    def __init__(
        self,
        screening_data: str | pd.DataFrame,
        dob_data: str | pd.DataFrame,
        n_females: int | None = None,
    ):
        self.screening_data = (
            screening_data
            if isinstance(screening_data, pd.DataFrame)
            else pd.read_csv(screening_data)
        )
        self.dob_data = (
            dob_data if isinstance(dob_data, pd.DataFrame) else pd.read_csv(dob_data)
        )

        self.columns = settings.processing.column_names

        self.dob_map = self._create_dob_map()
        self.n_females = n_females

        self.processed_data = None
        self.X = None
        self.row_map = None

    def _create_dob_map(self):
        """Method to map PIDs to birthdates.

        Only including people with 'B (bosatt)' status.
        """
        return (
            self.dob_data[self.dob_data[self.columns.dob.name] == "B"]
            .set_index(self.columns.pid)
            .to_dict()[self.columns.dob.date]
        )

    def _add_dob_column(self, df):
        df[self.columns.dob.date] = df[self.columns.pid].map(self.dob_map)
        return df

    def _convert_to_datetime(self, df, columns):
        def datetime_conversion(x):
            return pd.to_datetime(x, format="%d.%m.%Y")

        df[columns] = df[columns].apply(datetime_conversion)

    def _add_age_column(self, df):
        time_cols = self.columns.get_date_columns()
        self._convert_to_datetime(df, columns=time_cols)
        for col in (self.columns.cyt.date, self.columns.hist.date):
            df.loc[df[col].notna(), "age"] = (
                df[self.columns.cyt.date] - df[self.columns.dob.date]
            ).apply(lambda x: x.days)

    def _add_risk_column(self, df):
        for screening in settings.processing.risk_maps.keys():
            df["risk"] = df[screening].map(settings.processing.risk_maps[screening])

    def _gather_information(self, df):
        df = self._add_dob_column(df)
        self._add_age_column(df)
        self._add_risk_column(df)
        return df

    def _exclude_invalid_data(self, df: pd.DataFrame, min_n_tests: int = 2):
        new_df = df.dropna()
        person_counts = new_df[self.columns.pid].value_counts()
        rejected_pids = person_counts[person_counts.values < min_n_tests].index
        return new_df[~new_df[self.columns.pid].isin(rejected_pids)]

    def prepare_data(self) -> pd.DataFrame:
        df = pd.DataFrame.copy(
            self.screening_data[self.columns.get_screening_columns()]
        )

        if self.n_females is not None:
            n_total = df[self.columns.pid].nunique()
            max_n_females = self.n_females if self.n_females <= n_total else n_total

            selected = np.random.choice(
                self.screening_data[self.columns.pid].unique(),
                size=max_n_females,
                replace=False,
            )
            selected.sort()
            df = df[df[self.columns.pid].isin(selected)]

        df = self._gather_information(df)
        return self._exclude_invalid_data(df)

    def _find_n_bins(self, df):
        age_min, age_max = df["age"].min(), df["age"].max()
        avg_days_per_month = 30.437

        def ceildiv(a, b):
            return int(-(a // -b))

        return ceildiv(
            age_max - age_min,
            settings.processing.months_per_timepoint * avg_days_per_month,
        )

    def _assign_age_bins(self, df, n_bins):
        _, bin_edges = np.histogram(df["age"], bins=n_bins)
        bin_edges[0] -= 1  # to make sure the youngest age is included
        indexes = np.arange(n_bins)
        df["bin"] = pd.cut(df["age"], bins=bin_edges, labels=indexes)

    def _assign_row_index(self, df: pd.DataFrame):
        individuals = sorted(df["PID"].unique())
        n_females = len(individuals)
        row_map = dict(zip(individuals, np.arange(n_females)))
        df["row"] = df[self.columns.pid].map(row_map)
        return row_map

    def _create_age_aligned_matrix(self, df: pd.DataFrame):
        n_bins = self._find_n_bins(df)
        self._assign_age_bins(df, n_bins)
        row_map = self._assign_row_index(df)

        X = np.zeros([len(row_map), n_bins])
        risks = df["risk"].to_numpy()
        X[df.row, df.bin] = risks

        return X, row_map

    def generate_observation_matrix(self):
        self.processed_data = self.prepare_data()
        self.X, self.row_map = self._create_age_aligned_matrix(self.processed_data)
        return self.X

    def to_csv(self, location, **kwargs):
        if self.processed_data is None:
            self.prepare_data()
        self.processed_data.to_csv(location, **kwargs)
