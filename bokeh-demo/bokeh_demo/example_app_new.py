"""Example Bokeh server app, taken from docs.

See https://docs.bokeh.org/en/latest/docs/user_guide/server.html"""


from random import random
import pathlib
import itertools

from bokeh.layouts import column, row
from bokeh.models import Button, ColumnDataSource, Circle, CDSView, IndexFilter
from bokeh.models.callbacks import CustomJS
from bokeh.palettes import RdYlBu3
from bokeh.plotting import figure, curdoc
from matfact.data_generation import Dataset
import pandas as pd

def get_permutation_list(array):
    return [i for i, v in sorted(enumerate(array), key=lambda iv: iv[1])]

dataset_path = pathlib.Path('..') / "data/dataset1"
dataset = Dataset.from_file(dataset_path)
# Fake deltas
deltas = [random() * 2 - 1 for _ in range(dataset.X.shape[0])]
permutations = get_permutation_list(deltas)

xs = list(itertools.repeat(list(range(dataset.X.shape[1])), dataset.X.shape[0]))
ys = dataset.X.tolist()
source = ColumnDataSource({"xs": xs, "ys": ys, "x": list(range(dataset.X.shape[0])), "y": [deltas[i] for i in permutations]})
line_view = CDSView(source=source, filters=[])



def print_attr(attr, old, new):
    print(f"{attr} changed from {old} to {new}")
    if attr == "indices":
        source.selected.indices = new
        source.selected.indices = new
        line_view.filters = [IndexFilter(new)] if new else []

# add a button widget and configure with the call back
button = Button(label="Press Me")
# button.on_click(callback)

# create a plot and style its properties
delta_figure = figure(tools="tap,box_select")
delta_scatter = delta_figure.circle(source=source)

log_figure = figure(tools="tap,lasso_select")
lines=log_figure.multi_line(xs="xs", ys="ys", source=source, view=line_view)


source.selected.on_change("indices", print_attr)
# source.selected.on_change("multiline_indices", print_attr)


# put the button and plot in a layout and add to the document
curdoc().add_root(
    row(
        column(button, delta_figure),
        log_figure,
    )
)