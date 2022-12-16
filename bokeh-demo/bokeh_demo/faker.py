from collections import defaultdict
from typing import Sequence

import numpy as np

from .exam_data import EXAM_RESULT_MAPPING, ExamResult


def get_inverse_mapping() -> dict[int, list[ExamResult]]:
    possible_diagnosis = defaultdict(list)
    for type, states in EXAM_RESULT_MAPPING.items():
        for diagnosis_index, coarse_state in enumerate(states):
            possible_diagnosis[coarse_state].append(ExamResult(type, diagnosis_index))
    return dict(possible_diagnosis)  # We want KeyError for unknown states


class Faker:
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed=seed)
        self.coarse_state_to_exam_result = get_inverse_mapping()

    def get_fake_year_of_birth(
        self, person_index: int, first_possible: float = 1970, spread: float = 30
    ) -> float:
        """Generate a fake date of birth.

        NB. does not return the same date for a given index. This could
        be fixed by adding some sort of memory."""
        return first_possible + self.rng.random() * spread

    def get_fake_vaccine_age(
        self,
        vaccine_start_year: float = 12.0,
        vaccine_spread: float = 10,
        vaccine_prob: float = 0.3,
    ) -> None | float:
        # Beta 2, 5 is centered around 0.2 with a steep falloff.
        return (
            vaccine_start_year + vaccine_spread * self.rng.beta(2, 5)
            if self.rng.random() < vaccine_prob
            else None
        )

    def get_fake_detailed_result(
        self, coarse_exam_result: Sequence[int]
    ) -> Sequence[ExamResult | None]:
        return [
            self.rng.choice(self.coarse_state_to_exam_result[state]) if state else None
            for state in coarse_exam_result
        ]


faker = Faker()
