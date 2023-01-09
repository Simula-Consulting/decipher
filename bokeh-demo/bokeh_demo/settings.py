import pathlib

from pydantic import BaseSettings, DirectoryPath


class Settings(BaseSettings):
    number_of_epochs: int = 100
    dataset_path: DirectoryPath = pathlib.Path(__file__).parents[1] / "data/dataset1"

    label_map: list[str] = ["", "Normal", "Low risk", "High risk", "Cancer"]
    colors: list[str] = ["#027CE0", "#D9E019", "#E02019", "#E37E0B"]
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
