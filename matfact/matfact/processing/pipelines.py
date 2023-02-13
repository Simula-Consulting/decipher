from sklearn.pipeline import Pipeline

from matfact.processing.transformers import (
    AgeAdder,
    AgeBinAssigner,
    BirthdateAdder,
    DataSampler,
    DatetimeConverter,
    InvalidRemover,
    RiskAdder,
    RowAssigner,
)
from matfact.settings import settings

matfact_pipeline = Pipeline(
    [
        ("birthdate_adder", BirthdateAdder()),
        ("datetime_converter", DatetimeConverter()),
        ("age_adder", AgeAdder()),
        ("risk_adder", RiskAdder()),
        ("invalid_remover", InvalidRemover()),
        (
            "data_sampler",
            DataSampler(max_n_females=settings.processing.max_n_females),
        ),
        ("age_bin_assigner", AgeBinAssigner()),
        (
            "row_assigner",
            RowAssigner(save_path=settings.processing.row_map_save_location),
        ),
    ]
)
