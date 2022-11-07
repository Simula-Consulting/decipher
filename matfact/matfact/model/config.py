from pydantic import BaseModel

from matfact import settings


class ModelConfig(BaseModel):
    shift_budget: list[int] = []
    # W: npt.NDArray
    # D: npt.NDArray
    # J: npt.NDArray
    # K: npt.NDArray

    lambda1: float = 1.0
    lambda2: float = 1.0
    lambda3: float = 1.0

    iter_U: int = 2
    iter_V: int = 2

    learning_rate: float = 0.001
    number_of_states: int = settings.default_number_of_states
