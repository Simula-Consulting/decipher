from dataclasses import dataclass
from pathlib import Path
from enum import Enum

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

from matfact.processing.transformers import BirthdateAdder


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
    NORMAL_w_blood = "Normal m betennelse eller blod"
    NORMAL_wo_cylinder = "Normal uten sylinder"
    ADC = "ADC"
    SCC = "SCC"
    ACIS = "ACIS"
    ASC_US = "ASC-US"
    Nonconclusive = "Uegnet"  # HPV has a diagnosis called 'uegned' (lowercase)
    #
    METASTASIS = "Metastase"
    CANCER = "Cancer Cervix cancer andre/usp"
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


def read_raw_df(screening_data_path: Path, dtypes: dict | None = None, datetime_cols: list | None = None):
    dtypes = dtypes or {
        "cytMorfologi": "category",
        "histMorfologi": "Int64",
        "hpvResultat": "category",
    }
    datetime_cols = datetime_cols or ["hpvDate", "cytDate", "histDate"]

    return pd.read_csv(
        screening_data_path,
        dtype=dtypes,
        parse_dates=datetime_cols,
        dayfirst=True,
    )

def exam_pipeline() -> Pipeline:
    return Pipeline([
        ("cleaner", CleanData()),
        ("birthdate_adder", BirthdateAdder()),
        ("wide_to_long", ToExam()),
        ("age_adder", AgeAdder(date_field="exam_date", birth_field="FOEDT")),
    ],
    verbose=True
    )

class CleanData(BaseEstimator, TransformerMixin):
    dtypes = {
        "cytMorfologi": "category",
        "histMorfologi": "Int64",
        "hpvResultat": "category",
        "risk": "Int64",
    }
    def fit(self, X, y=None):
        for column, dtype in X.dtypes.items():
            if column in self.dtypes and (expected_type:=self.dtypes[column]) != dtype:
                raise ValueError(f"Column {column} must have dtype {expected_type}, but it is {dtype}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X

class ToExam(BaseEstimator, TransformerMixin):
    def __init__(self, fields_to_keep: list | None = None) -> None:
        self.fields_to_keep = fields_to_keep or ["PID", "FOEDT"]
        super().__init__()


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
            .astype({"exam_type": "category"})
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
        exams["exam_diagnosis"] = (
            exams["exam_diagnosis"]
            .astype("str")
            .apply(lambda diagnosis_string: Diagnosis(diagnosis_string))
            .astype("category")
        )

        return exams.join(X[self.fields_to_keep], on="index")

    @staticmethod
    def _map_exam_type(field_name) -> ExamTypes:
        return {
            "cytDate": ExamTypes.Cytology,
            "histDate": ExamTypes.Histology,
            "hpvDate": ExamTypes.HPVCobas,
        }[field_name]

class AgeAdder(BaseEstimator, TransformerMixin):
    def __init__(self, date_field: str, birth_field: str, age_field: str = "age") -> None:
        self.date_field = date_field
        self.birth_field = birth_field
        self.age_field = age_field
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.age_field] = X[self.date_field] - X[self.birth_field]
        return X

class ExtractPeople(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        def _minmax(column):
            return (min(column), max(column))

        person_df = (
            X.groupby("PID")
            .agg(
                {
                    "exam_date": _minmax,
                    "age": _minmax,
                    "FOEDT": "first",  # We assume all FOEDT are the same
                }
            )
            .reset_index()  # We want PID as explicit column
        )

        # Rename columns
        # person_df.columns = ["_".join(x) for x in person_df.columns]  # Flatten names
        person_df = person_df.rename(
            columns={
                "exam_date": "lexis_line_endpoints_year",
                "age": "lexis_line_endpoints_age",
            }
        )

        # Add some auxillary columns
        person_df["lexis_line_endpoints_person_index"] = person_df["PID"].transform(
            lambda pid: (pid, pid)
        )


        # Dummies
        person_df["exam_results"] = [[0]] * len(person_df.index)
        person_df["exam_time_age"] = [[0]] * len(person_df.index)
        person_df["prediction_time"] = 0
        person_df["predicted_exam_result"] = 0
        person_df["delta"] = 0
        person_df["vaccine_age"] = None
        person_df["vaccine_year"] = None
        person_df["vaccine_type"] = None
        person_df["vaccine_line_endpoints_age"] = [[]] * len(person_df.index)
        person_df["vaccine_line_endpoints_year"] = [[]] * len(person_df.index)
        return person_df


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
