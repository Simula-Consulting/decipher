import pytest


def test_an_exception() -> None:
    with pytest.raises(ZeroDivisionError):
        5 / 0
