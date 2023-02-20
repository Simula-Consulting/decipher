from pydantic import BaseSettings


class Settings(BaseSettings):
    number_of_epochs: int = 100

    label_map: list[str] = ["", "Normal", "Low risk", "High risk", "Cancer"]
    color_palette: list[str] = ["#13D697", "#0B6BB3", "#FFB60A", "#EF476F"]
    vaccine_line_color: str = "rgba(153, 153, 255, 0.5)"
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
