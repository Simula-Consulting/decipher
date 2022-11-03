# type: ignore
# flake8: noqa
"""MVP of the interactive frontend, with mocked alternative models."""


import itertools
import pathlib
from dataclasses import dataclass, fields
from random import random
from typing import Any, Type

import numpy as np
import pandas as pd
from bokeh.layouts import column, row
from bokeh.models import (
    Button,
    CDSView,
    Circle,
    ColumnDataSource,
    DataTable,
    HoverTool,
    IndexFilter,
    TableColumn,
)
from bokeh.models.callbacks import CustomJS
from bokeh.models.tickers import FixedTicker
from bokeh.palettes import RdYlBu3
from bokeh.plotting import curdoc, figure
from bokeh.transform import linear_cmap
from matfact.data_generation import Dataset
from matfact.experiments import train_and_log
from matfact.experiments.logging import dummy_logger_context
from matfact.plotting.diagnostic import _calculate_delta


def get_permutation_list(array):
    return [i for i, v in sorted(enumerate(array), key=lambda iv: iv[1])]


def get_x_pred(p_pred):
    return np.argmax(p_pred, axis=1) + 1


@dataclass
class Model:
    name: str
    ys: Any
    deltas: list[float]
    predicted: list[int]
    probabilities: Any
    x_true: list[int]
    t_pred: list[int]

    @classmethod
    def mock_from_real(cls, name: str, real_model: "Model"):
        probabilities = cls.perturb_probabilities(real_model.probabilities)
        predicted = np.argmax(probabilities, axis=1) + 1
        deltas = _calculate_delta(probabilities, real_model.x_true)
        ys = cls.update_ys(real_model.ys, predicted, real_model.t_pred)

        return cls(
            name=name,
            ys=ys,
            deltas=deltas,
            predicted=predicted,
            probabilities=probabilities,
            x_true=real_model.x_true,
            t_pred=real_model.t_pred,
        )

    def to_dict(self):
        return {
            f"{self.name}_{field.name}": getattr(self, field.name)
            for field in fields(self)
            if field.name != "name"
        }

    @staticmethod
    def update_ys(ys, predicted, t_pred):
        new_ys = ys.copy()
        for i, _ in enumerate(new_ys):
            new_ys[i][t_pred[i]] = predicted[i]
        return new_ys

    @staticmethod
    def perturb_probabilities(probabilities, fac=0.3):
        new_probabilities = np.empty_like(probabilities)
        for i, probs in enumerate(probabilities):
            new = probs + (2 * np.random.random(4) - 1) * fac
            new += np.min(new)
            new = new / np.linalg.norm(new)
            new_probabilities[i] = new
        return new_probabilities


# Import data
dataset_path = pathlib.Path(__file__).parent.parent / "data/dataset1"
dataset = Dataset.from_file(dataset_path)
X_train, X_test, _, _ = dataset.get_split_X_M()

# Fit the model, predict on test set
output = train_and_log(
    X_train,
    X_test,
    use_threshold_optimization=False,
    logger_context=dummy_logger_context,
)

# Extract the quantities of interest from output dict
p_pred = output["meta"]["results"]["p_pred"]
x_pred = output["meta"]["results"]["x_pred"]
t_pred = output["meta"]["results"]["t_pred"]
x_true = output["meta"]["results"]["x_true"].astype(int)
valid_rows = output["meta"]["results"]["valid_rows"]
X_test = X_test[valid_rows]

xs = list(itertools.repeat(list(range(X_test.shape[1])), X_test.shape[0]))

ys = X_test.tolist()
ys_pred = X_test.copy()
ys_pred[range(len(ys_pred)), t_pred] = x_pred
ys_pred = ys_pred.tolist()
deltas = _calculate_delta(p_pred, x_true - 1)
model = Model(
    name="real",
    ys=ys_pred,
    deltas=deltas,
    predicted=x_pred,
    probabilities=p_pred,
    x_true=x_true,
    t_pred=t_pred,
)

fake_models = ["hmm", "super"]
models = [model] + [Model.mock_from_real(fake_name, model) for fake_name in fake_models]

permutations = get_permutation_list(model.deltas)
x = list(range(len(x_true)))
sorted_x = [permutations.index(i) for i in x]

# Set up the Bokeh data source
# Each row corresponds to one individual
models_dict = {}
for model in models:
    models_dict |= model.to_dict()
source = ColumnDataSource(
    {
        "xs": xs,
        "ys": ys,
        "x": x,
        "perm": sorted_x,
        "true": x_true,
    }
    | models_dict
)


## Set up Bokeh plots
default_tools = "pan,wheel_zoom,box_zoom,save,reset,help"

# Add the Delta score figure
delta_figure = figure(
    title="Delta score distribution",
    x_axis_label="Individual",
    y_axis_label="Delta score (lower better)",
    tools="tap,lasso_select," + default_tools,
)
delta_scatter = delta_figure.circle(
    x="perm",
    y="real_deltas",
    radius=0.3,
    fill_color=linear_cmap("real_deltas", "Spectral6", -1, 1),
    source=source,
)

# Add the time trajectory figure
log_figure = figure(
    title="Individual state trajectories",
    x_axis_label="Time",
    y_axis_label="State",
    tools="tap,lasso_select," + default_tools,
    y_range=(0, 4),
)
log_figure.yaxis.ticker = FixedTicker(ticks=[0, 1, 2, 3, 4])
log_figure.add_tools(
    HoverTool(
        tooltips=[
            ("Id", "$index"),
            ("Predict", "@predicted"),
            ("Probabilities", "@probabilities"),
        ]
    )
)
lines = log_figure.multi_line(
    xs="xs",
    ys="ys",
    source=source,
    legend_label="Actual observation",
    nonselection_line_alpha=0.0,
)
lines_pred = log_figure.multi_line(
    xs="xs",
    ys="real_ys",
    source=source,
    color="red",
    legend_label="Predicted",
    nonselection_line_alpha=0.0,
)

# Add the table over individuals
person_table = DataTable(
    source=source,
    columns=[
        TableColumn(title="Delta score", field="real_deltas"),
        TableColumn(title="Delta score ordering", field="perm"),
        TableColumn(title="Predicted state", field="real_predicted"),
        TableColumn(title="Correct state", field="true"),
        # TableColumn(title="Prediction discrepancy", field="prediction_discrepancy"),
    ],
)

# Set up our event handler


# Put everything in the document
curdoc().add_root(
    # column(
    row(
        delta_figure,
        log_figure,
        person_table,
    ),
    # )
)
