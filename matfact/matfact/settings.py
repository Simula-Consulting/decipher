from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, BaseSettings, root_validator


class PathSettings(BaseModel):
    base = Path(__file__).parents[1]
    test = base / "tests"
    dataset = base / "datasets"
    results = base / "results"
    figure = results / "figures"

    create_default: bool = True  # Create artifact directories if non-existent


class MatFactSettings(BaseModel):
    number_of_states: int = 4
    age_segments: tuple[int, int] = (76, 116)
    weights: list[float] = []

    @root_validator(allow_reuse=True)
    def set_default_weights(cls, values):
        if len(values["weights"]) == 0:
            values["weights"] = range(1, values["number_of_states"] + 1)
        return values


class ObservationMatrixGenerationSettings(BaseModel):
    observation_probabilities: list[float] = [0.01, 0.03, 0.08, 0.12, 0.04]
    minimum_number_of_observations = 3
    sparsity_level: int = 6
    confidence_parameter: float = 2.5
    memory_length: int = 5

    # Matrix dimensions
    rank: int = 5
    n_rows: int = 1000
    n_columns: int = 50


class GaussianGeneratorSettings(BaseModel):
    """Values used to generate a screening dataset with a
    Discrete Gaussian Distribution (DGD).

    Choices result from research and is explained in Mikal Stapnes'
    Masters thesis (page 21-22).
    """

    scale_factor: float = 3.0
    kernel_param: float = 5e-4
    centre_minmax: tuple[float, float] = (70, 170)


class CensoringSettings(BaseModel):
    """Shape parameters for the beta-binomial used to censor generated data."""

    a: float = 4.57
    b: float = 5.74


class PropensityWeightSettings(BaseModel):
    tau: float = 1.0
    gamma: float = 3.0


class ConvergenceMonitorSettings(BaseModel):
    number_of_epochs: int = 2000
    epochs_per_val: int = 5
    patience: int = 200


@dataclass
class ColumnName:
    name: str
    date: str


@dataclass
class ColumnNameCollection:
    pid: str

    cyt: ColumnName
    hist: ColumnName
    dob: ColumnName

    def get_date_columns(self):
        return [self.cyt.date, self.hist.date, self.dob.date]

    def get_screening_columns(self):
        return [
            self.pid,
            self.cyt.name,
            self.cyt.date,
            self.hist.name,
            self.hist.date,
        ]


class DataProcessingSettings(BaseModel):

    risk_maps: dict[str, dict[Any, int]] = {
        "cytMorfologi": {
            "Normal": 1,
            "LSIL": 2,
            "ASC-US": 2,
            "ASC-H": 3,
            "HSIL": 3,
            "ACIS": 3,
            "AGUS": 3,
            "SCC": 4,
            "ADC": 4,
        },
        "histMorfologi": {
            10: 1,
            100: 1,
            1000: 1,
            8001: 1,
            74006: 2,
            74007: 3,
            74009: 3,
            80021: 1,
            80032: 3,
            80402: 3,
            80703: 4,
            80833: 4,
            81403: 4,
            82103: 4,
        },
    }

    months_per_timepoint: int = 3
    dateformat: str = "%d.%m.%Y"

    # Personal identifier
    pid: str = "PID"

    # Screening data column names
    cyt = ColumnName(name="cytMorfologi", date="cytDate")
    hist = ColumnName(name="histMorfologi", date="histDate")

    # Date of birth data column names
    dob = ColumnName(name="STATUS", date="FOEDT")

    column_names = ColumnNameCollection(pid=pid, cyt=cyt, hist=hist, dob=dob)

    # processing pipeline configuration
    min_n_tests: int = 2
    max_n_females: int | None = None
    row_map_save_location: str | None = None

    # data files
    raw_screening_data_path: str = "screening_data.csv"
    raw_dob_data_path: str = "dob_data.csv"


class Settings(BaseSettings):
    paths = PathSettings()
    matfact_defaults = MatFactSettings(_env_file=".env")
    matrix_generation = ObservationMatrixGenerationSettings()
    propensity_weights = PropensityWeightSettings()
    convergence = ConvergenceMonitorSettings()
    gauss_gen = GaussianGeneratorSettings()
    censoring = CensoringSettings()
    processing = DataProcessingSettings()

    class Config:
        env_nested_delimiter = "__"


settings = Settings(_env_file=".env")
