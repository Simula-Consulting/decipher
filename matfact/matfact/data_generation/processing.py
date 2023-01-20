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

        self.n_females = n_females

        self.processed_data: pd.DataFrame | None = None
        self.X: np.ndarray | None = None
        self.dob_map: dict[int, str] | None = None
        self.row_map: dict[int, int] | None = None

    def _create_dob_map(self) -> dict[int, str]:
        """Method to map PIDs to birthdates.

        Only including people with 'B (bosatt)' status.
        """
        return (
            self.dob_data[self.dob_data[self.columns.dob.name] == "B"]
            .set_index(self.columns.pid)
            .to_dict()[self.columns.dob.date]
        )

    def _add_dob_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Method to add date of birth do a dataframe based on a map from person id."""
        self.dob_map = self._create_dob_map()
        df[self.columns.dob.date] = df[self.columns.pid].map(self.dob_map)
        return df

    def _convert_to_datetime(self, df: pd.DataFrame, columns: list[str]) -> None:
        """Method to convert string values to datetimes."""

        def datetime_conversion(x: str) -> pd.Timestamp:
            return pd.to_datetime(x, format=settings.processing.dateformat)

        df[columns] = df[columns].apply(datetime_conversion)

    def _add_age_column(self, df: pd.DataFrame) -> None:
        """Method to add a column with person age at screening time."""
        time_cols = self.columns.get_date_columns()
        self._convert_to_datetime(df, columns=time_cols)
        for col in (self.columns.cyt.date, self.columns.hist.date):
            df.loc[df[col].notna(), "age"] = (
                df[self.columns.cyt.date] - df[self.columns.dob.date]
            ).apply(lambda x: x.days)

    def _add_risk_column(self, df: pd.DataFrame) -> None:
        """Method to add a column with diagnosis as risk levels."""
        # TODO: this is bugged, it overwrites values to NaN
        # TODO: also include test to ensure this doesn't happen
        for screening in settings.processing.risk_maps.keys():
            df["risk"] = df[screening].map(settings.processing.risk_maps[screening])

    def _gather_information(self, df: pd.DataFrame) -> pd.DataFrame:
        """Method to prepare the dataframe by gathering all available information."""
        df = self._add_dob_column(df)
        self._add_age_column(df)
        self._add_risk_column(df)
        return df

    def _exclude_invalid_data(
        self, df: pd.DataFrame, min_n_tests: int = 2
    ) -> pd.DataFrame:
        """Method to exlude invalid data by dropping columns with NaN values
        or females with only 1 screening."""
        new_df = df.dropna(subset=["age", "risk"])
        person_counts = new_df[self.columns.pid].value_counts()
        rejected_pids = person_counts[person_counts.values < min_n_tests].index
        return new_df[~new_df[self.columns.pid].isin(rejected_pids)]

    def prepare_data(self) -> pd.DataFrame:
        """Method to create a processed DataFrame containing only relevant and valid data."""
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

    def _find_n_bins(self, df: pd.DataFrame) -> int:
        """Method to calculate the number of age-bins required,
        based on the age span in the data and the desired number of months per bin."""
        age_min, age_max = df["age"].min(), df["age"].max()
        avg_days_per_month = 30.437

        def ceildiv(a: float, b: float) -> int:
            """Function to perform ceiling division, opposite of floor division."""
            return int(-(a // -b))

        try:
            return ceildiv(
                age_max - age_min,
                settings.processing.months_per_timepoint * avg_days_per_month,
            )
        except ValueError:
            from IPython import embed

            embed()

    def _assign_age_bins(self, df: pd.DataFrame, n_bins: int) -> None:
        """Method to assign results into age bins (columns) based on the age at time of screening."""
        _, bin_edges = np.histogram(df["age"], bins=n_bins)
        bin_edges[0] -= 1  # to make sure the youngest age is included
        indexes = np.arange(n_bins)
        df["bin"] = pd.cut(df["age"], bins=bin_edges, labels=indexes)

    def _assign_row_index(self, df: pd.DataFrame) -> dict[int, int]:
        """Method to assign a unique row to every person.
        Also returns this mapping for future reference."""
        individuals = sorted(df["PID"].unique())
        n_females = len(individuals)
        row_map = dict(zip(individuals, np.arange(n_females)))
        df["row"] = df[self.columns.pid].map(row_map)
        return row_map

    def _create_age_aligned_matrix(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, dict[int, int]]:
        """Method to turn a processed Dataframe into a numpy array."""
        n_bins = self._find_n_bins(df)
        self._assign_age_bins(df, n_bins)
        row_map = self._assign_row_index(df)

        X = np.zeros([len(row_map), n_bins])
        risks = df["risk"].to_numpy()
        X[df.row, df.bin] = risks

        return X, row_map

    def generate_observation_matrix(self) -> np.ndarray:
        """Method to process the screening data and produce an age-aligned observation matrix."""
        self.processed_data = self.prepare_data()
        if len(self.processed_data) < 1:
            return []
        self.X, self.row_map = self._create_age_aligned_matrix(self.processed_data)
        return self.X

    def to_csv(self, location: str, **kwargs) -> None:
        """Method to save the processed data to a .csv file."""
        if self.processed_data is None:
            self.prepare_data()
        self.processed_data.to_csv(location, **kwargs)  # type: ignore
