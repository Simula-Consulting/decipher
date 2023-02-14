from pathlib import Path

from pydantic import BaseModel, BaseSettings, root_validator
from decipher.data_generation.settings import (
    ObservationMatrixGenerationSettings,
    CensoringSettings,
    GaussianGeneratorSettings,
    HMMGeneratorSettings,
)


class PathSettings(BaseModel):
    base = Path(__file__).parents[2]
    test = base / "tests"
    dataset = base / "datasets"
    results = base / "results"
    figure = results / "figures"

    create_default: bool = True  # Create artifact directories if non-existent


class MatFactSettings(BaseModel):
    number_of_states: int = 4
    age_segments: tuple[int, int] = (76, 116)
    weights: list[float] = []

    @root_validator
    def set_default_weights(cls, values):
        if len(values["weights"]) == 0:
            values["weights"] = range(1, values["number_of_states"] + 1)
        return values


class PropensityWeightSettings(BaseModel):
    tau: float = 1.0
    gamma: float = 3.0


class ConvergenceMonitorSettings(BaseModel):
    number_of_epochs: int = 2000
    epochs_per_val: int = 5
    patience: int = 200


class Settings(BaseSettings):
    paths = PathSettings()
    matfact_defaults = MatFactSettings(_env_file=".env")
    matrix_generation = ObservationMatrixGenerationSettings()
    propensity_weights = PropensityWeightSettings()
    convergence = ConvergenceMonitorSettings()
    gauss_gen = GaussianGeneratorSettings()
    hmm_gen = HMMGeneratorSettings()
    censoring = CensoringSettings()

    class Config:
        env_nested_delimiter = "__"


settings = Settings(_env_file=".env")
