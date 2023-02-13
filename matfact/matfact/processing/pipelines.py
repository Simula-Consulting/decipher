from sklearn.pipeline import Pipeline

from matfact.processing.transformers import (
    AgeAdder,
    AgeBinAssigner,
    BirthdateAdder,
    DatetimeConverter,
    InvalidRemover,
    RiskAdder,
    RowAssigner,
    SampleFemales,
)

matfact_pipeline = Pipeline(
    [
        ("birthdate_adder", BirthdateAdder()),
        ("datetime_converter", DatetimeConverter()),
        ("age_adder", AgeAdder()),
        ("risk_adder", RiskAdder()),
        ("invalid_remover", InvalidRemover()),
        ("female_sampler", SampleFemales(max_n_females=None)),
        ("age_bin_assigner", AgeBinAssigner()),
        ("row_assigner", RowAssigner()),
    ]
)
