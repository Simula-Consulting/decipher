# type: ignore
# flake8: noqa
"""Example Bokeh server app, taken from docs.

See https://docs.bokeh.org/en/latest/docs/user_guide/server.html"""


import itertools
import pathlib
from random import random

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


dataset_path = pathlib.Path(__file__).parent.parent / "data/dataset1"
dataset = Dataset.from_file(dataset_path)
X_train, X_test, _, _ = dataset.get_split_X_M()
output = train_and_log(
    X_train,
    X_test,
    use_threshold_optimization=False,
    logger_context=dummy_logger_context,
)
p_pred = output["meta"]["results"]["p_pred"]
x_pred = output["meta"]["results"]["x_pred"]
t_pred = output["meta"]["results"]["t_pred"]
x_true = output["meta"]["results"]["x_true"].astype(int)
valid_rows = output["meta"]["results"]["valid_rows"]
deltas = _calculate_delta(p_pred, x_true - 1)
X_test = X_test[valid_rows]


# Fake deltas
# deltas = [random() * 2 - 1 for _ in range(dataset.X.shape[0])]
permutations = get_permutation_list(deltas)
x = list(range(len(x_true)))
sorted_x = [permutations.index(i) for i in x]

xs = list(itertools.repeat(list(range(X_test.shape[1])), X_test.shape[0]))
ys = X_test.tolist()
ys_pred = X_test.copy()
ys_pred[range(len(ys_pred)), t_pred] = x_pred
ys_pred = ys_pred.tolist()
source = ColumnDataSource(
    {
        "xs": xs,
        "ys": ys,
        "ys_pred": ys_pred,
        "x": x,
        "y": deltas,
        "perm": sorted_x,
        "predicted": x_pred,
        "true": x_true,
        "prediction_discrepancy": np.abs(x_pred - x_true),
        "probabilities": [[f"{ps:0.2f}" for ps in lst] for lst in p_pred],
    }
)
line_view = CDSView(source=source, filters=[])


def print_attr(attr, old, new):
    print(f"{attr} changed from {old} to {new}")
    if attr == "indices":
        source.selected.indices = new
        line_view.filters = [IndexFilter(new)] if new else []


default_tools = "pan,wheel_zoom,box_zoom,save,reset,help"
# create a plot and style its properties
delta_figure = figure(
    title="Delta score distribution",
    x_axis_label="Individual",
    y_axis_label="Delta score (lower better)",
    tools="tap,lasso_select," + default_tools,
)
delta_scatter = delta_figure.circle(
    x="perm", radius=0.3, fill_color=linear_cmap("y", "Spectral6", -1, 1), source=source
)

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
    xs="xs", ys="ys", source=source, view=line_view, legend_label="Actual observation"
)
lines_pred = log_figure.multi_line(
    xs="xs",
    ys="ys_pred",
    source=source,
    view=line_view,
    color="red",
    legend_label="Predicted",
)

person_table = DataTable(
    source=source,
    columns=[
        TableColumn(title="Delta score", field="y"),
        TableColumn(title="Delta score ordering", field="perm"),
        TableColumn(title="Predicted state", field="predicted"),
        TableColumn(title="Correct state", field="true"),
        TableColumn(title="Prediction discrepancy", field="prediction_discrepancy"),
    ],
)


source.selected.on_change("indices", print_attr)
# source.selected.on_change("multiline_indices", print_attr)


# put the button and plot in a layout and add to the document
curdoc().add_root(
    # column(
    row(
        delta_figure,
        log_figure,
        person_table,
    ),
    # )
)
