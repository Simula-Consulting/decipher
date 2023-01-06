from pathlib import Path

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, BaseSettings


class PathSettings(BaseModel):
    base: Path = Path(__file__).parents[1]
    figure: Path = base / "figures"


class StaticSettings(BaseModel):
    """Static definitions/results based on research."""

    age_partitions: npt.NDArray[np.int_] = np.array(
        [16, 20, 25, 30, 35, 40, 50, 60, 100]
    )

    # Initial state probabilities: age group x probability initial state (adjusted to row
    # stochastic).
    p_init_states: npt.NDArray[np.float_] = np.array(
        [
            [0.92992, 0.06703, 0.00283, 0.00022],
            [0.92881, 0.06272, 0.00834, 0.00013],
            [0.93408, 0.04945, 0.01632, 0.00015],
            [0.94918, 0.03554, 0.01506, 0.00022],
            [0.95263, 0.03250, 0.01463, 0.00024],
            [0.95551, 0.03303, 0.01131, 0.00015],
            [0.96314, 0.02797, 0.00860, 0.00029],
            [0.96047, 0.02747, 0.01168, 0.00038],
        ]
    )

    # Transition intensities: age group x state transition.
    # fmt: off
    lambda_sr = np.array(
        [
            # N0->L1  L1->H2   H2->C3   L1->N0   H2->L1   N0->D4   L1->D4   H2->D4   C3->D4
            [0.02027, 0.01858, 0.00016, 0.17558, 0.24261, 0.00002, 0.00014, 0.00222, 0.01817],
            [0.01202, 0.02565, 0.00015, 0.15543, 0.11439, 0.00006, 0.00016, 0.00082, 0.03024],
            [0.00746, 0.03959, 0.00015, 0.14622, 0.07753, 0.00012, 0.00019, 0.00163, 0.03464],
            [0.00584, 0.04299, 0.00055, 0.15576, 0.07372, 0.00012, 0.00016, 0.00273, 0.04211],
            [0.00547, 0.03645, 0.00074, 0.15805, 0.06958, 0.00010, 0.00015, 0.00398, 0.04110],
            [0.00556, 0.02970, 0.00127, 0.17165, 0.08370, 0.00010, 0.00014, 0.00518, 0.03170],
            [0.00440, 0.02713, 0.00161, 0.19910, 0.10237, 0.00020, 0.00029, 0.00618, 0.02772],
            [0.00403, 0.03826, 0.00419, 0.24198, 0.06951, 0.00104, 0.00115, 0.02124, 0.02386],
        ]
    )
    # fmt: on

    class Config:
        arbitrary_types_allowed = True


class Settings(BaseSettings):
    paths = PathSettings()
    static = StaticSettings()

    class Config:
        env_nested_delimiter = "__"


settings = Settings(_env_file=".env")
