import pathlib

from pydantic import BaseModel, BaseSettings, DirectoryPath


class DataPaths(BaseModel):
    base_path: DirectoryPath = (
        pathlib.Path(__file__).parents[2] / "processing" / "tests" / "test_datasets"
    )
    screening_data_path: pathlib.Path = base_path / "test_screening_data.csv"
    dob_data_path: pathlib.Path = base_path / "test_dob_data.csv"


class Settings(BaseSettings):
    number_of_epochs: int = 100
    dataset_path: DirectoryPath = pathlib.Path(__file__).parents[1] / "data/dataset1"
    transfer_path: DirectoryPath = pathlib.Path(__file__).parents[1] / "data/transfer/"
    selected_pids_path: pathlib.Path = transfer_path / "selected_pids.json"

    data_paths: DataPaths = DataPaths()

    label_map: dict[int | float, str] = {
        1: "Normal",
        2: "Low risk",
        3: "High risk",
        4: "Cancer",
        float("nan"): "Unknown",
    }
    color_palette: dict[int | float, str] = {
        1: "#13D697",
        2: "#0B6BB3",
        3: "#FFB60A",
        4: "#EF476F",
        float("nan"): "gray",
    }
    """Color palette for the different risk levels."""
    default_tools: list[str] = [
        "pan",
        "wheel_zoom",
        "box_zoom",
        "save",
        "reset",
        "help",
        "examine",  # For debugging
    ]
    extra_tools: list[str] = ["tap", "lasso_select"]
    range_padding: float = 0.1


settings = Settings()
