from dataclasses import dataclass
from enum import Enum


class ExamTypes(str, Enum):
    Cytology = "cytology"
    Histology = "histology"
    HPV = "HPV"


@dataclass
class ExamResult:
    type: ExamTypes
    result: int  # Must be looked up


EXAM_RESULT_LOOKUP = {
    ExamTypes.Cytology: [
        "CytDiagnosis0",
        "CytDiagnosis1",
        "CytDiagnosis2",
    ],
    ExamTypes.Histology: [
        "HistDiagnosis0",
        "HistDiagnosis1",
        "HistDiagnosis2",
        "HistDiagnosis3",
        "HistDiagnosis4",
    ],
    ExamTypes.HPV: [
        "HPV Negative",
        "HPV Positive",
    ],
}

# Mapping from diagnosis to coarse state
EXAM_RESULT_MAPPING = {
    ExamTypes.Cytology: [
        1,
        2,
        3,
    ],
    ExamTypes.Histology: [
        1,
        1,
        2,
        3,
        4,
    ],
    ExamTypes.HPV: [
        1,
        3,
    ],
}
