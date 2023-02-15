from sklearn.pipeline import Pipeline

from matfact.processing.transformers import (
    AgeAdder,
    AgeBinAssigner,
    BirthdateAdder,
    ColumnSelector,
    DataSampler,
    DatetimeConverter,
    InvalidRemover,
    RiskAdder,
    RowAssigner,
)


def matfact_pipeline(
    *,
    verbose: bool = True,
    columns: list[str] | None = None,
    birthday_file: str | None = None,
    min_n_tests: int | None = None,
    max_n_females: int | None = None,
    row_map_save_path: str | None = None,
):
    """Returns a sklearn type pipeline for processing the matfact screening data."""
    return Pipeline(
        [
            ("column_selector", ColumnSelector(columns=columns)),
            ("birthdate_adder", BirthdateAdder(birthday_file=birthday_file)),
            ("datetime_converter", DatetimeConverter()),
            ("age_adder", AgeAdder()),
            ("risk_adder", RiskAdder()),
            ("invalid_remover", InvalidRemover(min_n_tests=min_n_tests)),
            ("data_sampler", DataSampler(max_n_females=max_n_females)),
            ("age_bin_assigner", AgeBinAssigner()),
            ("row_assigner", RowAssigner(row_map_save_path=row_map_save_path)),
        ],
        verbose=verbose,
    )
