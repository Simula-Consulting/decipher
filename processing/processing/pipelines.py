from pathlib import Path

from sklearn.pipeline import Pipeline

from processing.transformers import (
    AgeAdder,
    AgeBinAssigner,
    DataSampler,
    DatetimeConverter,
    FolkeregInfoAdder,
    InvalidRemover,
    RiskAdder,
    RiskAdderHHMM,
    RowAssigner,
    TestIndexAdder,
    ToExam
)


def matfact_pipeline(
    *,
    verbose: bool = True,
    birthday_file: Path | None = None,
    min_n_tests: int | None = None,
    max_n_females: int | None = None,
    row_map_save_path: Path | None = None,
):
    """Returns a sklearn type pipeline for processing the matfact screening data."""
    return Pipeline(
        [
            (
                "birthdate_adder",
                FolkeregInfoAdder(birthday_file=birthday_file, death_column=False),
            ),
            ("datetime_converter", DatetimeConverter()),
            ("age_adder", AgeAdder(in_years=False)),
            ("risk_adder", RiskAdder()),
            ("invalid_remover", InvalidRemover(min_n_tests=min_n_tests)),
            ("data_sampler", DataSampler(max_n_females=max_n_females)),
            ("age_bin_assigner", AgeBinAssigner()),
            ("row_assigner", RowAssigner(row_map_save_path=row_map_save_path)),
        ],
        verbose=verbose,
    )


def HHMM_pipeline(
    *,
    verbose: bool = True,
    birthday_file: Path | None = None,
):
    """Returns a sklearn type Pipeline for processing data for HHMM."""
    return Pipeline(
        [
            (
                "folkereg_adder",
                FolkeregInfoAdder(birthday_file=birthday_file, death_column=True),
            ),
            ("to_exam_converter", ToExam(fields_to_keep=["PID", "FOEDT", "is_dead"])),
            ("datetime_converter", DatetimeConverter(columns=["exam_date"])),
            ("age_adder", AgeAdder(in_years=True, target_columns=["exam_date"])),
            ("risk_adder", RiskAdderHHMM()),
            ("test_index", TestIndexAdder()),
            ("invalid_remover", InvalidRemover(min_n_tests=0)),
        ],
        verbose=verbose
    )
