from dataclasses import dataclass
from enum import Enum

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class VaccineType(str, Enum):
    VaccineType1 = "Vaccine type 1"
    VaccineType2 = "Vaccine type 2"


class ExamTypes(str, Enum):
    Cytology = "cytology"
    Histology = "histology"
    HistologyTreatment = "histology treatment"
    HPVCobas = "HPV Cobas"
    HPVGenXpert = "HPV genXpert"


class Diagnosis(str, Enum):
    CytDiagnosis0 = "CytDiagnosis0"
    CytDiagnosis1 = "CytDiagnosis1"
    CytDiagnosis2 = "CytDiagnosis2"
    HistDiagnosis0 = "HistDiagnosis0"
    HistDiagnosis1 = "HistDiagnosis1"
    HistDiagnosis2 = "HistDiagnosis2"
    HistDiagnosis3 = "HistDiagnosis3"
    HistDiagnosis4 = "HistDiagnosis4"
    #
    AGUS = "AGUS"
    LSIL = "LSIL"
    HSIL = "HSIL"
    ASC_H = "ASC-H"
    NORMAL = "Normal"
    ADC = "ADC"
    SCC = "SCC"
    ACIS = "ACIS"
    ASC_US = "ASC-US"
    #
    Hist10 = "10"
    Hist100 = "100"
    Hist1000 = "1000"
    Hist8001 = "8001"
    Hist74006 = "74006"
    Hist74007 = "74007"
    Hist74009 = "74009"
    Hist80021 = "80021"
    Hist80032 = "80032"
    Hist80402 = "80402"
    Hist80703 = "80703"
    Hist80833 = "80833"
    Hist81403 = "81403"
    Hist82103 = "82103"
    # HPV common
    HPVNegative = "negativ"
    HPVPositive = "positiv"
    HPVUnknown = "uegnet"
    # HPV cobas
    HPVCobas16 = "HPV 16"
    HPVCobas18 = "HPV 18"
    HPVCobasChannel12 = "Cobas Channel 1"
    """Channel collecting 31, 33, 35, 39, 45, 51, 52, 56, 58, 59, 66, and 68"""
    # HPV genXpert
    HPVGenXpert16 = "HPV 16"
    HPVGenXpert18_45 = "HPV pool 18/45"
    HPVGenXpertchannel1 = "genXpert channel 1"
    """Channel collecting 31, 33, 35, 52, 58; 51, 59; 39, 56, 66, 68"""


class ToExam(BaseEstimator, TransformerMixin):
    fields_to_keep = ["PID", "FOEDT", "age", "risk", "bin", "row"]

    def fit(self, X, y=None):
        self._has_hpv_data = "hpvDate" in X
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        mapper = {"cytDate": "cytMorfologi", "histDate": "histMorfologi"}
        if self._has_hpv_data:
            mapper["hpvDate"] = "hpvResultat"

        # Transform from wide to long
        exams = (
            X.reset_index()
            .melt(
                id_vars="index",
                value_vars=mapper.keys(),
                var_name="exam_type",
                value_name="exam_date",
            )
            .dropna()
        )

        # Join on result columns
        exams = exams.join(X[mapper.values()], on="index")

        # Ugly clean hist
        exams["histMorfologi"] = exams["histMorfologi"].astype("Int64")

        # Add result column
        conditions = [exams["exam_type"] == key for key in mapper]
        values = [exams[key] for key in mapper.values()]
        exams["exam_diagnosis"] = np.select(conditions, values)

        # Drop the raw exam result
        exams = exams.drop(columns=mapper.values())

        # Remap exam types
        exams["exam_type"] = exams["exam_type"].transform(self._map_exam_type)
        exams["exam_diagnosis"] = exams["exam_diagnosis"].astype('str').transform(lambda diagnosis_string: Diagnosis(diagnosis_string))

        return exams.join(X[self.fields_to_keep], on="index")

    @staticmethod
    def _map_exam_type(field_name) -> ExamTypes:
        return {
            "cytDate": ExamTypes.Cytology,
            "histDate": ExamTypes.Histology,
            "hpvDate": ExamTypes.HPVCobas,
        }[field_name]


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
    ExamTypes.HPVCobas: [
        Diagnosis.HPVNegative,
        Diagnosis.HPVCobas16,
        Diagnosis.HPVCobas18,
        Diagnosis.HPVCobasChannel12,
    ],
    ExamTypes.HPVGenXpert: [
        Diagnosis.HPVNegative,
        Diagnosis.HPVGenXpert16,
        Diagnosis.HPVGenXpert18_45,
        Diagnosis.HPVGenXpertchannel1,
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
    Diagnosis.HPVCobas16: 3,
    Diagnosis.HPVCobas18: 3,
    Diagnosis.HPVCobasChannel12: 2,
    Diagnosis.HPVGenXpert16: 3,
    Diagnosis.HPVGenXpert18_45: 3,
    Diagnosis.HPVGenXpertchannel1: 2,
}
