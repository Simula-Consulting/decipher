# type: ignore
# flake8: noqa
"""Example Bokeh server app, taken from docs.

See https://docs.bokeh.org/en/latest/docs/user_guide/server.html"""


import itertools
import pathlib
from random import random

import pandas as pd
from bokeh.layouts import column, row
from bokeh.models import Button, CDSView, Circle, ColumnDataSource, IndexFilter
from bokeh.models.callbacks import CustomJS
from bokeh.palettes import RdYlBu3
from bokeh.plotting import curdoc, figure
from matfact.data_generation import Dataset
from matfact.experiments import train_and_log
from matfact.experiments.logging import dummy_logger_context
from matfact.plotting.diagnostic import _calculate_delta


def get_permutation_list(array):
    return [i for i, v in sorted(enumerate(array), key=lambda iv: iv[1])]


dataset_path = pathlib.Path(__file__).parent.parent / "data/dataset1"
dataset = Dataset.from_file(dataset_path)
X_train, X_test, _, _ = dataset.get_split_X_M()
output = train_and_log(X_train, X_test)  # , logger_context=dummy_logger_context)
p_pred = output["meta"]["results"]["p_pred"]
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
source = ColumnDataSource({"xs": xs, "ys": ys, "x": x, "y": deltas, "perm": sorted_x})
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
delta_scatter = delta_figure.circle(x="perm", source=source)

log_figure = figure(
    title="Individual state trajectories",
    x_axis_label="Time",
    y_axis_label="State",
    tools="tap,lasso_select," + default_tools,
)
lines = log_figure.multi_line(xs="xs", ys="ys", source=source, view=line_view)


source.selected.on_change("indices", print_attr)
# source.selected.on_change("multiline_indices", print_attr)


# put the button and plot in a layout and add to the document
curdoc().add_root(
    row(
        delta_figure,
        log_figure,
    )
)
