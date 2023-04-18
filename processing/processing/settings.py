from pydantic import BaseConfig, BaseModel
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
    dob_status: ColumnName

    def get_date_columns(self):
        return [self.cyt.date, self.hist.date, self.dob.date, self.dob_status.date]

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
        "hpvResultat": {
            "negativ": 0,
            "positiv": 1,
            "uegnet": 0,
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
    dob_status = ColumnName(name="STATUS", date="STATUSDATE")

    column_names = ColumnNameCollection(
        pid=pid, cyt=cyt, hist=hist, dob=dob, dob_status=dob_status
    )
    # processing pipeline configuration
    min_n_tests: int = 2
    max_n_females: int | None = None
    row_map_save_location: str | None = None

    # data files
    file_location: Path = Path("../processing/tests/test_datasets")
    raw_screening_data_path: Path = file_location / "test_screening_data.csv"
    raw_dob_data_path: Path = file_location / "test_dob_data.csv"


class Settings(BaseConfig):
    processing = DataProcessingSettings()


settings = Settings()
