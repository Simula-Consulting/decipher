from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt

from matfact.settings import settings


class FactorizerType(Enum):
    SNMF = "SklearnNMF"
    SDL = "SklearnDictionaryLearning"
    STSVD = "SklearnTruncatedSVD"
    DMF = "DecipherMF"


class Hyperparams(Enum):
    lambda1 = "lambda1"
    lambda2 = "lambda2"
    rank = "rank"


@dataclass
class SklearnFactorizerConfig:
    """Configuration class for the Sklearn Factorizer model."""

    model_type: FactorizerType = FactorizerType.SNMF
    lambda1: float = 1.0
    lambda2: float = 1.0
    rank: int = 5

    number_of_states: int = settings.matfact_defaults.number_of_states


@dataclass
class ExperimentConfig:
    """Configuration class for the Sklearn Factorizer model."""

    # DatasetConfig
    N: int = 1000
    T: int = 100
    sparsity: int = 100

    # HyperparameterConfig
    model_type: FactorizerType = FactorizerType.DMF
    lambda1: npt.NDArray[np.float64] = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    lambda2: npt.NDArray[np.float64] = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    rank: npt.NDArray[np.int_] = np.array([5])
    iterating_hyperparameters: tuple[Hyperparams, Hyperparams] = (
        Hyperparams.lambda1,
        Hyperparams.lambda2,
    )
    number_of_states: int = settings.matfact_defaults.number_of_states
