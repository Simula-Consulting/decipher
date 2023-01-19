"""Test the factorizers CMF, WCMF, and SCMF"""

import numpy as np
import numpy.typing as npt
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays
from pytest import approx

from matfact.model import CMF, SCMF, WCMF, BaseMF
from matfact.model.config import IdentityWeighGetter, ModelConfig, WeightGetter

identity_config = ModelConfig(
    shift_budget=[0],
    weight_matrix_getter=IdentityWeighGetter(),
    minimal_value_matrix_getter=np.zeros,
    rank=2,
)


def array2D(
    dtype=int,
    element_strategy=st.integers,
    min_value=0,
    max_value=5,
    min_side=1,
    max_side=None,
) -> st.SearchStrategy[np.ndarray]:
    return arrays(
        dtype=dtype,
        shape=array_shapes(
            min_dims=2, max_dims=2, min_side=min_side, max_side=max_side
        ),
        elements=element_strategy(min_value=min_value, max_value=max_value),
    ).filter(lambda arr: np.all(np.sum(arr, axis=1)))


@st.composite
def model_config_strategy(
    draw,
    max_shift: int = 0,
    max_rank: int = 5,
    weight_matrix_getter: WeightGetter | None = None,
) -> ModelConfig:
    weight_matrix_getter = weight_matrix_getter or IdentityWeighGetter()
    shift = draw(st.integers(min_value=0, max_value=max_shift))
    rank = draw(st.integers(min_value=1, max_value=max_rank))
    return ModelConfig(
        shift_budget=list(range(-shift, shift + 1)),
        weight_matrix_getter=weight_matrix_getter,
        minimal_value_matrix_getter=draw(st.sampled_from((np.ones, np.zeros))),
        rank=rank,
    )


class SimpleWeightGetter(WeightGetter):
    """Simple WeightGetter returning the observation matrix itself."""

    def __call__(self, observation_matrix: npt.NDArray[np.int_]) -> npt.NDArray:
        return observation_matrix.copy()


@settings(deadline=None)
@given(array2D(), model_config_strategy(weight_matrix_getter=SimpleWeightGetter()))
def test_special_case_wcmf_scmf_equal(
    observation_matrix: npt.NDArray[np.int_], config: ModelConfig
) -> None:
    """Test that WCMF and SCMF behave the same for no shift."""
    wcmf = WCMF(observation_matrix, config)
    scmf = SCMF(observation_matrix, config)

    assert wcmf.U == approx(scmf.U)
    assert wcmf.V == approx(scmf.V)

    number_of_steps = 2
    for _ in range(number_of_steps):
        assert wcmf.loss() == approx(scmf.loss())
        wcmf.U = wcmf._approx_U()
        scmf.U = scmf._approx_U()
        assert wcmf.U == approx(scmf.U)

        wcmf._update_V()
        scmf._update_V()
        assert wcmf.V == approx(scmf.V)


@given(array2D(), st.integers(min_value=0, max_value=3))
def test_loss_equal_initial(
    observation_matrix: npt.NDArray[np.int_], max_shift: int
) -> None:
    """Test that SCMF and WCMF have the same initial loss value.

    Note that this holds true even for shifts, as the shift is initialized to 0."""

    config = ModelConfig(
        shift_budget=list(range(-max_shift, max_shift + 1)),
        weight_matrix_getter=IdentityWeighGetter(),
        minimal_value_matrix_getter=np.zeros,
        initial_basic_profiles_getter=lambda a, b: np.ones((a, b)),
        rank=2,
    )

    class BaseLossSCMF(SCMF):
        """SCMF with V regularization computed from non-padded matrices.

        SCMF internally uses matrices that are padded according to the shift
        budget, also for loss. As the loss is currently not normalized by dimensions,
        this causes the padded system to have a higher loss for V L2 and
        convolutional regularization terms. The reconstruction loss is not affected,
        as we there use the observation mask.
        This class uses only the non-padded V to compute the L2 and convolutional loss,
        to be able to meaningfully compare with WCMF."""

        def V_L2_loss_term(self) -> float:
            return self.config.lambda2 * np.square(
                np.linalg.norm(self.V - self.J[self.Ns : -self.Ns or None, :])
            )

        def temporal_loss_term(self) -> float:
            return self.config.lambda3 * np.square(
                np.linalg.norm(
                    self.KD[self.Ns : -self.Ns or None, self.Ns : -self.Ns or None]
                    @ self.V
                )
            )

    wcmf = WCMF(observation_matrix, config=config)
    scmf = BaseLossSCMF(observation_matrix, config=config)

    assert wcmf.loss() == approx(scmf.loss())


@settings(deadline=None)
@given(
    array2D(), model_config_strategy(max_shift=3), st.sampled_from((CMF, WCMF, SCMF))
)
def test_factorizer_x_invariant(
    observation_matrix: npt.NDArray[np.int_],
    config: ModelConfig,
    factorizer_class: type[BaseMF],
) -> None:
    """Test that the factorizer's X attribute is invariant."""
    X = observation_matrix.copy()
    factorizer = factorizer_class(observation_matrix, config)
    assert np.all(factorizer.X == X)

    number_of_steps = 3
    for _ in range(number_of_steps):
        factorizer.run_step()
        assert np.all(factorizer.X == X)


@settings(deadline=None)
@given(array2D(), st.sampled_from((CMF, WCMF, SCMF)), model_config_strategy())
def test_output_array_shape(
    observation_matrix: npt.NDArray[np.int_],
    factorizer_class: type[BaseMF],
    config: ModelConfig,
):
    """Test that the factorizer's matrix shapes are correct."""
    number_of_samples, number_of_time_steps = observation_matrix.shape
    factorizer = factorizer_class(observation_matrix, config)
    correct_shapes = {
        "X": (number_of_samples, number_of_time_steps),
        "V": (number_of_time_steps, config.rank),
        "U": (number_of_samples, config.rank),
    }
    # We first to one null-operation, to check that the initial matrices are correct
    for operation in (lambda: None, factorizer.run_step):
        operation()

        for matrix_name, shape in correct_shapes.items():
            assert (
                getattr(factorizer, matrix_name).shape == shape
            ), f"{matrix_name} should have shape {shape}, however it is {getattr(factorizer, matrix_name).shape}"
