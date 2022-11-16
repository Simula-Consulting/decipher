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
    CustomJSExpr,
    DataTable,
    Div,
    HoverTool,
    IndexFilter,
    Legend,
    LegendItem,
    Slider,
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
deltas = _calculate_delta(p_pred, x_true - 1)
X_test = X_test[valid_rows]

permutations = get_permutation_list(deltas)
x = list(range(len(x_true)))
sorted_x = [permutations.index(i) for i in x]

xs = list(itertools.repeat(list(range(X_test.shape[1])), X_test.shape[0]))
ys = X_test.tolist()
ys_pred = X_test.copy()
ys_pred[range(len(ys_pred)), t_pred] = x_pred
ys_pred = ys_pred.tolist()

number_of_individuals = len(xs)
number_of_time_steps = len(xs[0])

rng = np.random.default_rng()
max_offset = 40  # Years
fake_date_of_birth = [
    1960 + max_offset * factor for factor in rng.random(number_of_individuals)
]


# Set up the Bokeh data source
# Each row corresponds to one individual
source = ColumnDataSource(
    {
        "is": [[i] * len(xs[0]) for i in range(len(xs))],  # List of indices, hack!
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

years = ((age + fake_date_of_birth[i] for age in ages) for i, ages in enumerate(xs))

scatter_source = ColumnDataSource(
    {
        key: list(value)
        for key, value in {
            "x": itertools.chain.from_iterable(xs),
            "y": itertools.chain.from_iterable(ys),
            "i": itertools.chain.from_iterable(
                (
                    itertools.repeat(i, number_of_time_steps)
                    for i in range(number_of_individuals)
                )
            ),
            "year": itertools.chain.from_iterable(years),
        }.items()
    }
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
    x="perm", radius=0.3, fill_color=linear_cmap("y", "Spectral6", -1, 1), source=source
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
    ys="ys_pred",
    source=source,
    color="red",
    legend_label="Predicted",
    nonselection_line_alpha=0.0,
)

# Add Lexis-ish plot
lexis_ish_figure = figure(
    title="Lexis-ish plot", tools="tap,lasso_select," + default_tools
)  # , x_range=(0,100), y_range=(0,40))
markers = (None, "square", "circle", "diamond")
colors = [None, "blue", "green", "red"]


def cycle_mapper(cycle):
    return {
        "expr": CustomJSExpr(
            args={"markers": cycle}, code="return this.data.y.map(i => markers[i]);"
        )
    }


lexis_scatter = lexis_ish_figure.scatter(
    "x",
    "i",
    marker=cycle_mapper(markers),
    color=cycle_mapper(colors),
    source=scatter_source,
    legend_group="y",
)


# Add Lexis plot
lexis_figure = figure(
    title="Lexis-ish plot", tools="tap,lasso_select," + default_tools
)  # , x_range=(0,100), y_range=(0,40))
markers = (None, "square", "circle", "diamond")
colors = [None, "blue", "green", "red"]


def cycle_mapper(cycle):
    return {
        "expr": CustomJSExpr(
            args={"markers": cycle}, code="return this.data.y.map(i => markers[i]);"
        )
    }


lexis_scatter = lexis_figure.scatter(
    "x",
    "year",
    marker=cycle_mapper(markers),
    color=cycle_mapper(colors),
    source=scatter_source,
    legend_group="y",
)


def select_person(attr, old, selected_people):
    all_indices = [
        i
        for i, person_index in enumerate(scatter_source.data["i"])
        if person_index in selected_people
    ]

    scatter_source.selected.indices = all_indices
    source.selected.indices = selected_people


def set_group_selected(attr, old, new):
    selected_people = list({scatter_source.data["i"][i] for i in new})
    select_person(None, None, selected_people)


scatter_source.selected.on_change("indices", set_group_selected)
source.selected.on_change("indices", select_person)

# Add the table over individuals
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

slider = Slider(start=1, end=20, step=0.5, value=10)
slider.js_link("value", lexis_scatter.glyph, "size")
# for line in lexis_lines:
#     slider.js_link("value", line.glyph, "size")

information_box = Div(
    text="""<h1>Welcome to the Simula Consulting interactive exploration demo!</h1>
    <br/><br/>

    <div style="border: 1px solid rgba(0,0,0,0.125); padding: 10px; border-radius: 5px; width: 50%;">
    Hei

    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed hendrerit nec elit quis consectetur. Cras in viverra quam. Donec vel dictum arcu. Suspendisse at mauris lobortis, efficitur magna sit amet, tristique nulla. Duis molestie mi vel consequat vestibulum. Pellentesque mollis aliquet lorem non varius. Sed placerat enim urna, in accumsan dui pharetra id. Donec ultricies convallis eleifend. Proin et nisl libero. Maecenas ultricies arcu tortor. Phasellus quis ante nisi. Aenean dictum est lorem, eu fermentum nibh mollis non.
    </div>
    """,
)

# Put everything in the document
curdoc().add_root(
    column(
        information_box,
        row(
            delta_figure,
            column(lexis_figure, lexis_ish_figure, slider),
            log_figure,
            person_table,
        ),
    )
)
