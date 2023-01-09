from dataclasses import dataclass
from enum import Enum


class ExamTypes(str, Enum):
    Cytology = "cytology"
    Histology = "histology"
    HistologyTreatment = "histology treatment"
    HPV = "HPV"


class Diagnosis(str, Enum):
    CytDiagnosis0 = "CytDiagnosis0"
    CytDiagnosis1 = "CytDiagnosis1"
    CytDiagnosis2 = "CytDiagnosis2"
    HistDiagnosis0 = "HistDiagnosis0"
    HistDiagnosis1 = "HistDiagnosis1"
    HistDiagnosis2 = "HistDiagnosis2"
    HistDiagnosis3 = "HistDiagnosis3"
    HistDiagnosis4 = "HistDiagnosis4"
    HPVNegative = "HPV Negative"
    HPVPositive = "HPV Positive"


@dataclass
class ExamResult:
    type: ExamTypes
    result: Diagnosis


EXAM_RESULT_LOOKUP = {
    ExamTypes.Cytology: [
        Diagnosis.CytDiagnosis0,
        Diagnosis.CytDiagnosis1,
        Diagnosis.CytDiagnosis2,
    ],
    ExamTypes.Histology: [
        Diagnosis.HistDiagnosis0,
        Diagnosis.HistDiagnosis1,
        Diagnosis.HistDiagnosis2,
        Diagnosis.HistDiagnosis3,
        Diagnosis.HistDiagnosis4,
    ],
    ExamTypes.HistologyTreatment: [
        Diagnosis.HistDiagnosis0,
        Diagnosis.HistDiagnosis1,
        Diagnosis.HistDiagnosis2,
        Diagnosis.HistDiagnosis3,
        Diagnosis.HistDiagnosis4,
    ],
    ExamTypes.HPV: [
        Diagnosis.HPVNegative,
        Diagnosis.HPVPositive,
    ],
}

# Mapping from diagnosis to coarse state
EXAM_RESULT_MAPPING = {
    Diagnosis.CytDiagnosis0: 1,
    Diagnosis.CytDiagnosis1: 2,
    Diagnosis.CytDiagnosis2: 3,
    Diagnosis.HistDiagnosis0: 1,
    Diagnosis.HistDiagnosis1: 1,
    Diagnosis.HistDiagnosis2: 2,
    Diagnosis.HistDiagnosis3: 3,
    Diagnosis.HistDiagnosis4: 4,
    Diagnosis.HPVNegative: 1,
    Diagnosis.HPVPositive: 3,
}
