from pathlib import Path

from pydantic import BaseModel, BaseSettings


class PathSettings(BaseModel):
    base: Path = Path(__file__).parents[1]
    figure: Path = base / "figures"


class Settings(BaseSettings):
    paths = PathSettings()

    class Config:
        env_nested_delimiter = "__"


settings = Settings(_env_file=".env")
