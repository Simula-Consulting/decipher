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


class ObservationMatrixGenerationSettings(BaseModel):
    observation_probabilities: npt.NDArray[np.float_] = np.array(
        [0.01, 0.03, 0.08, 0.12, 0.04]
    )
    minimum_number_of_observations = 3
    sparsity_level: int = 6

    # Matrix dimensions
    rank: int = 5
    n_rows: int = 1000
    n_columns: int = 50

    class Config:
        arbitrary_types_allowed = True  # Allow validation of numpy array


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


class Settings(BaseSettings):
    paths = PathSettings()
    matfact_defaults = MatFactSettings()
    matrix_generation = ObservationMatrixGenerationSettings()
    propensity_weights = PropensityWeightSettings()
    convergence = ConvergenceMonitorSettings()
    gauss_gen = GaussianGeneratorSettings()
    censoring = CensoringSettings()


settings = Settings()
