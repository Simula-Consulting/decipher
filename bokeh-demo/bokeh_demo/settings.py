import pathlib

from pydantic import BaseSettings, DirectoryPath


class Settings(BaseSettings):
    number_of_epochs: int = 100
    dataset_path: DirectoryPath = pathlib.Path(__file__).parents[1] / "data/dataset1"
    transfer_path: DirectoryPath = pathlib.Path(__file__).parents[1] / "data/transfer/"
    selected_pids_path: pathlib.Path = transfer_path / "selected_pids.pkl"

    label_map: list[str] = ["", "Normal", "Low risk", "High risk", "Cancer"]
    color_palette: list[str] = ["#13D697", "#0B6BB3", "#FFB60A", "#EF476F"]
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
