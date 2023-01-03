from pathlib import Path

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, BaseSettings


class PathSettings(BaseModel):
    base = Path(__file__).parents[1]
    test = base / "tests"
    dataset = base / "datasets"
    results = base / "results"
    figure = results / "figures"

    create_default: bool = True  # Create artifact directories if non-existent


class MatFactSettings(BaseModel):
    number_of_states: int = 4
    weights: range = range(1, number_of_states + 1)
    age_segments: tuple = (76, 116)

    class Config:
        arbitrary_types_allowed = True  # Allow validation of range


class DataGenerationSettings(BaseModel):
    observation_probabilities: npt.NDArray[np.float_] = np.array(
        [0.01, 0.03, 0.08, 0.12, 0.04]
    )
    minimum_number_of_observations = 3

    class Config:
        arbitrary_types_allowed = True  # Allow validation of numpy array


class PropensityWeightSettings(BaseModel):
    tau: float = 1.0
    gamma: float = 3.0


class ConvergenceMonitorSettings(BaseModel):
    number_of_epochs: int = 2000
    epochs_per_val: int = 5
    patience: int = 200


class Settings(BaseSettings):
    paths = PathSettings()
    matfact_defaults = MatFactSettings()
    data_generation = DataGenerationSettings()
    propensity_weights = PropensityWeightSettings()
    convergence = ConvergenceMonitorSettings()


settings = Settings()
