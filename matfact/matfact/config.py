from pathlib import Path

from pydantic import BaseModel, BaseSettings, root_validator


class PathSettings(BaseModel):
    base = Path(__file__).parents[1]
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


class ObservationMatrixGenerationSettings(BaseModel):
    observation_probabilities: list[float] = [0.01, 0.03, 0.08, 0.12, 0.04]
    minimum_number_of_observations = 3
    sparsity_level: int = 6
    confidence_parameter: float = 2.5
    memory_length: int = 5

    # Matrix dimensions
    rank: int = 5
    n_rows: int = 1000
    n_columns: int = 50


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
    matfact_defaults = MatFactSettings(_env_file=".env")
    matrix_generation = ObservationMatrixGenerationSettings()
    propensity_weights = PropensityWeightSettings()
    convergence = ConvergenceMonitorSettings()
    gauss_gen = GaussianGeneratorSettings()
    censoring = CensoringSettings()

    class Config:
        env_nested_delimiter = "__"


settings = Settings(_env_file=".env")
