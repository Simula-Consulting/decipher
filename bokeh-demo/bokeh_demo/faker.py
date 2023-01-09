from collections import defaultdict
from typing import Iterable, Mapping, Sequence, TypeVar, overload

import numpy as np

from .exam_data import EXAM_RESULT_LOOKUP, EXAM_RESULT_MAPPING, ExamResult, VaccineType

T = TypeVar("T")
S = TypeVar("S")


@overload
def _invert_dict(input_dict: Mapping[T, Iterable[S]]) -> dict[S, list[T]]:
    ...


@overload
def _invert_dict(input_dict: Mapping[T, S]) -> dict[S, list[T]]:
    ...


def _invert_dict(input_dict):
    """Invert a dict, i.e. give a dict with value as keys and keys as values."""
    inverted = defaultdict(list)
    for key, values in input_dict.items():
        try:
            for value in values:
                inverted[value].append(key)
        except TypeError:  # Not iterable
            inverted[values].append(key)
    return dict(inverted)


COARSE_STATE_TO_DIAGNOSIS = _invert_dict(EXAM_RESULT_MAPPING)


def coarse_to_exam_result() -> dict[int, list[ExamResult]]:
    """Give a mapping from coarse states to possible ExamResults."""
    diagnosis_to_exam_types = _invert_dict(EXAM_RESULT_LOOKUP)
    mapping = defaultdict(list)
    for state, possible_diagnoses in COARSE_STATE_TO_DIAGNOSIS.items():
        for diagnosis in possible_diagnoses:
            for exam_type in diagnosis_to_exam_types[diagnosis]:
                mapping[state].append(ExamResult(type=exam_type, result=diagnosis))
    return dict(mapping)


class Faker:
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed=seed)
        self.coarse_state_to_exam_result = coarse_to_exam_result()

    def get_fake_year_of_birth(
        self, person_index: int, first_possible: float = 1970, spread: float = 30
    ) -> float:
        """Generate a fake date of birth.

        NB. does not return the same date for a given index. This could
        be fixed by adding some sort of memory."""
        return first_possible + self.rng.random() * spread

    def get_fake_vaccine_age(
        self,
        vaccine_start_age: float = 12.0,
        vaccine_spread: float = 10,
        vaccine_prob: float = 0.3,
    ) -> None | float:
        # Beta 2, 5 is centered around 0.2 with a steep falloff.
        return (
            vaccine_start_age + vaccine_spread * self.rng.beta(2, 5)
            if self.rng.random() < vaccine_prob
            else None
        )

    def get_fake_vaccine_type(self) -> VaccineType:
        # Ugly hack as rng.choice seems to convert the type to a string...
        return VaccineType(self.rng.choice([v.value for v in VaccineType]))

    def get_fake_detailed_result(
        self, coarse_exam_result: Sequence[int]
    ) -> Sequence[ExamResult | None]:
        return [
            self.rng.choice(self.coarse_state_to_exam_result[state]) if state else None
            for state in coarse_exam_result
        ]


faker = Faker()
