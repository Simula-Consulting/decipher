from dataclasses import dataclass
from enum import Enum

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


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
    # HPV common
    HPVNegative = "HPV negative"
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


class CreatePlottingData(BaseEstimator, TransformerMixin):
    """Takes in the exams dataframe and returns another dataframe for plotting Lexis plots."""

    def __init__(self, *, prediction_data: None = None):
        self.prediction_data = prediction_data

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        def _minmax(column):
            return (min(column), max(column))

        plotting_df = (
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
        # plotting_df.columns = ["_".join(x) for x in plotting_df.columns]  # Flatten names
        plotting_df = plotting_df.rename(
            columns={
                "exam_date": "lexis_line_endpoints_year",
                "age": "lexis_line_endpoints_age",
            }
        )

        # Add some auxillary columns
        plotting_df["lexis_line_endpoints_person_index"] = plotting_df["PID"].transform(
            lambda pid: (pid, pid)
        )


        # Dummies
        plotting_df["exam_results"] = plotting_df["PID"].map({pid: sub_df["risk"].to_list() for pid, sub_df in X.groupby("PID")})
        plotting_df["exam_time_age"] = plotting_df["PID"].map({pid: (sub_df["age"] / pd.Timedelta(days=365)).to_list() for pid, sub_df in X.groupby("PID")})
        plotting_df["vaccine_age"] = None
        plotting_df["vaccine_year"] = None
        plotting_df["vaccine_type"] = None
        plotting_df["vaccine_line_endpoints_age"] = [[]] * len(plotting_df.index)
        plotting_df["vaccine_line_endpoints_year"] = [[]] * len(plotting_df.index)

        if self.prediction_data is None:
            plotting_df["prediction_time"] = 0
            plotting_df["predicted_exam_result"] = 0
            plotting_df["delta"] = 0
        else:
            # TODO: Add real prediction data
            pass

        return plotting_df


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
