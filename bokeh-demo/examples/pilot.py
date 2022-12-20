# type: ignore
"""Example Bokeh server app.

The app consist of two main "parts"
  1. The data must be processed and put into sources
  2. The visualization itself

For part 1, the most important classes are `PredictionData` and `Person`.
`PredictionData` is responsible for reading in the data and then constructing
`Person` instances from it.
The `Person` class is responsible for generating the source objects to be used
by the visualization.
"""

import tensorflow as tf
from bokeh.layouts import column, row
from bokeh.models import CustomJS, RadioButtonGroup, RangeSlider, Switch, Toggle
from bokeh.plotting import curdoc
from matfact.data_generation import Dataset
from matfact.model.config import ModelConfig
from matfact.model.factorization.convergence import ConvergenceMonitor
from matfact.model.matfact import MatFact
from matfact.model.predict.dataset_utils import prediction_data

from bokeh_demo.backend import PredictionData, SourceManager
from bokeh_demo.frontend import (
    DeltaScatter,
    HistogramPlot,
    LexisPlot,
    LexisPlotAge,
    PersonTable,
    TrajectoriesPlot,
)
from bokeh_demo.settings import settings

tf.config.set_visible_devices([], "GPU")


# Import data
dataset = Dataset.from_file(settings.dataset_path)


def extract_and_predict(
    dataset: Dataset, model: MatFact | None = None
) -> PredictionData:
    model = model or MatFact(
        ModelConfig(
            epoch_generator=ConvergenceMonitor(
                number_of_epochs=settings.number_of_epochs
            ),
        )
    )

    X_train, X_test, _, _ = dataset.get_split_X_M()
    X_test_masked, t_pred_test, x_true_test = prediction_data(X_test)

    model.fit(X_train)
    predicted_probabilities = model.predict_probabilities(X_test, t_pred_test)
    # Could alternatively do matfact._predictor(predicted_probabilites) ...
    predicted_states = model.predict(X_test, t_pred_test)

    return PredictionData(
        X_train=X_train,
        X_test=X_test,
        X_test_masked=X_test_masked,
        time_of_prediction=t_pred_test,
        true_state_at_prediction=x_true_test,
        predicted_probabilities=predicted_probabilities,
        predicted_states=predicted_states,
    )


def example_app(source_manager):
    lp = LexisPlot(source_manager)
    lpa = LexisPlotAge(source_manager)
    delta = DeltaScatter(source_manager)
    traj = TrajectoriesPlot(source_manager)
    table = PersonTable(source_manager)
    hist = HistogramPlot(source_manager)
    filter_toggle = Switch(active=False)
    filter_toggle.on_change(
        "active", source_manager.filters["high_risk"].get_set_active_callback()
    )

    high_risk_toggle = Switch(active=False)
    high_risk_toggle.on_change(
        "active", source_manager.filters["high_risk_2"].get_set_active_callback()
    )

    high_risk_exam_toggle = Switch(active=False)
    high_risk_exam_toggle.on_change(
        "active", source_manager.filters["high_risk_3"].get_set_active_callback()
    )
    # radio_group = RadioButtonGroup(labels=["Off", "On", "Inverted"])
    # radio_group.on_change("active", source_manager.get_filter_callback("vaccine"))

    vaccine_filter_toggle = Switch(active=False)
    vaccine_filter_toggle.on_change(
        "active", source_manager.filters["vaccine_age"].get_set_active_callback()
    )
    vaccine_filter_invert_toggle = Switch(active=False)
    vaccine_filter_invert_toggle.on_change(
        "active", source_manager.filters["vaccine_age"].get_set_inverted_callback()
    )
    range_slider = RangeSlider(value=(0, 100), start=0, end=100)
    range_slider.on_change(
        "value", source_manager.filters["vaccine_age"].get_set_value_callback()
    )

    curdoc().add_root(
        column(
            row(
                lp.figure,
                lpa.figure,
                traj.figure,
                table.person_table,
            ),
            row(
                hist.figure,
                delta.figure,
                column(
                    filter_toggle,
                    high_risk_toggle,
                    high_risk_exam_toggle,
                    row(
                        vaccine_filter_toggle,
                        vaccine_filter_invert_toggle,
                        range_slider,
                    ),
                ),
            ),
        )
    )


def main():
    prediction_data = extract_and_predict(dataset)
    people = prediction_data.extract_people()
    source_manager = SourceManager.from_people(people)
    example_app(source_manager)


# Name is not main when run through bokeh serve, so no __name__ == __main__ guard
main()
