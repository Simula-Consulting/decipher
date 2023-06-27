import pathlib

from pydantic import BaseModel, BaseSettings, DirectoryPath


class DataPaths(BaseModel):
    base_path: DirectoryPath = (
        pathlib.Path(__file__).parents[2] / "processing" / "tests" / "test_datasets"
    )
    screening_data_path: pathlib.Path = base_path / "test_screening_data.csv"
    dob_data_path: pathlib.Path = base_path / "test_dob_data.csv"


class FeatureColumnNames(BaseModel):
    """Class to hold the names of the columns in the dataframes."""

    exam_date: str = "exam_date"
    age: str = "age"
    birthdate: str = "FOEDT"
    PID: str = "PID"
    risk: str = "risk"
    hpv_pos_count: str = "count_positive"
    risk_max: str = "risk_max"
    n_screenings: str = "number_of_screenings"
    age_last_exam: str = "age_last_exam"
    exam_type: str = "exam_type"
    hr_cytology: str = "high_risk_cytology"
    hr_histology: str = "high_risk_histology"
    exam_details: str = "exam_detailed_results"
    exam_diagnosis: str = "exam_diagnosis"


class Settings(BaseSettings):
    number_of_epochs: int = 100
    dataset_path: DirectoryPath = pathlib.Path(__file__).parents[1] / "data/dataset1"
    transfer_path: DirectoryPath = pathlib.Path(__file__).parents[1] / "data/transfer/"
    selected_pids_path: pathlib.Path = transfer_path / "selected_pids.json"

    data_paths: DataPaths = DataPaths()

    label_map: dict[int | None, str] = {
        1: "Normal",
        2: "Low risk",
        3: "High risk",
        4: "Cancer",
        None: "Unknown",
    }
    color_palette: dict[int | None, str] = {
        1: "#13D697",
        2: "#0B6BB3",
        3: "#FFB60A",
        4: "#EF476F",
        None: "gray",
    }
    """Color palette for the different risk levels."""
    default_tools: list[str] = [
        "pan",
        "wheel_zoom",
        "box_zoom",
        "save",
        "reset",
    ]
    extra_tools: list[str] = ["tap", "lasso_select"]
    range_padding: float = 0.1

    feature_column_names = FeatureColumnNames()


settings = Settings()
