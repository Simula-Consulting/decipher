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
from bokeh.layouts import row
from bokeh.plotting import curdoc
from matfact.data_generation import Dataset
from matfact.model.config import ModelConfig
from matfact.model.factorization.convergence import ConvergenceMonitor
from matfact.model.matfact import MatFact
from matfact.model.predict.dataset_utils import prediction_data

from bokeh_demo.backend import (
    PredictionData,
    link_sources,
    scatter_source_from_people,
    source_from_people,
)
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


def test_plot(person_source, exam_source):
    lp = LexisPlot(person_source, exam_source)
    lpa = LexisPlotAge(person_source, exam_source)
    delta = DeltaScatter(person_source, exam_source)
    traj = TrajectoriesPlot(person_source, exam_source)
    table = PersonTable(person_source, exam_source)
    hist = HistogramPlot(person_source, exam_source)

    curdoc().add_root(
        row(
            hist.figure,
            lp.figure,
            lpa.figure,
            delta.figure,
            traj.figure,
            table.person_table,
        ),
    )


def main():
    prediction_data = extract_and_predict(dataset)
    people = prediction_data.extract_people()
    person_source = source_from_people(people)
    exam_source = scatter_source_from_people(people)
    link_sources(person_source, exam_source)
    test_plot(person_source, exam_source)


main()
