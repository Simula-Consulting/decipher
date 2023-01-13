"""Test the factorizers CMF, WCMF, and SCMF"""

import numpy as np
import numpy.typing as npt
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays
from pytest import approx

from matfact.model import CMF, SCMF, WCMF, BaseMF
from matfact.model.config import IdentityWeighGetter, ModelConfig

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
def model_config_strategy(draw, max_shift: int = 0, max_rank: int = 5):
    shift = draw(st.integers(min_value=0, max_value=max_shift))
    rank = draw(st.integers(min_value=1, max_value=max_rank))
    return ModelConfig(
        shift_budget=list(range(-shift, shift + 1)),
        weight_matrix_getter=IdentityWeighGetter(),
        minimal_value_matrix_getter=draw(st.sampled_from((np.ones, np.zeros))),
        rank=rank,
    )


@settings(deadline=None)
@given(array2D(min_side=4))
def test_special_case_equal(observation_matrix: npt.NDArray[np.int_]) -> None:
    cmf = CMF(observation_matrix, identity_config)
    wcmf = WCMF(observation_matrix, identity_config)
    scmf = SCMF(observation_matrix, identity_config)

    for _ in range(10):
        cmf.run_step()
        wcmf.run_step()
    assert cmf.loss() == approx(wcmf.loss(), rel=1e-2)

    number_of_steps = 2
    for _ in range(number_of_steps):
        assert wcmf.loss() == approx(scmf.loss())
        wcmf.U = wcmf._approx_U()
        scmf.U = scmf._approx_U()
        assert wcmf.U == approx(scmf.U)

        wcmf._update_V()
        scmf._update_V()
        assert wcmf.V == approx(scmf.V)


@settings(database=None)
@given(array2D())
def test_special_case_wcmf_scmf_equal(observation_matrix: npt.NDArray[np.int_]) -> None:
    """Test that wcmf and scmf behave the same for no shift.

    TODO: Also include weights."""
    wcmf = WCMF(observation_matrix, identity_config)
    scmf = SCMF(observation_matrix, identity_config)

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
        rank=2,
    )
    wcmf = WCMF(observation_matrix, config=config)
    scmf = SCMF(observation_matrix, config=config)

    assert wcmf.loss() == approx(scmf.loss())


@settings(deadline=None)
@given(array2D(), st.integers(min_value=0, max_value=3))
def test_scmf_x_invariant(
    observation_matrix: npt.NDArray[np.int_], max_shift: int
) -> None:
    """Test that SCMF and WCMF have the same initial loss value.

    Note that this holds true even for shifts, as the shift is initialized to 0."""

    config = ModelConfig(
        shift_budget=list(range(-max_shift, max_shift + 1)),
        weight_matrix_getter=IdentityWeighGetter(),
        minimal_value_matrix_getter=np.zeros,
        rank=2,
    )

    X = observation_matrix.copy()
    scmf = SCMF(observation_matrix, config)
    assert np.all(scmf.X == X)

    number_of_steps = 3
    for _ in range(number_of_steps):
        scmf.run_step()
        assert np.all(scmf.X == X)


@settings(deadline=None)
@given(array2D(), st.sampled_from((CMF, WCMF, SCMF)), model_config_strategy())
def test_output_array_shape(
    observation_matrix: npt.NDArray[np.int_],
    factorizer_class: type[BaseMF],
    config: ModelConfig,
):
    number_of_samples, number_of_time_steps = observation_matrix.shape
    factorizer = factorizer_class(observation_matrix, config)
    correct_shapes = {
        "X": (number_of_samples, number_of_time_steps),
        "V": (number_of_time_steps, config.rank),
        "U": (number_of_samples, config.rank),
    }
    for operation in (lambda: None, factorizer.run_step):
        operation()

        for matrix_name, shape in correct_shapes.items():
            assert (
                getattr(factorizer, matrix_name).shape == shape
            ), f"{matrix_name} should have shape {shape}, however it is {getattr(factorizer, matrix_name).shape}"
