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

SC_logo_base64 = r"""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"""


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
    title="Lexis-ish plot",
    tools="tap,lasso_select," + default_tools,
    x_axis_label="Age at sample [months since 16]",
    y_axis_label="Individual #",
)
markers = (None, "square", "circle", "diamond")
colors = [None, "blue", "green", "red"]


def cycle_mapper(cycle):
    return {
        "expr": CustomJSExpr(
            args={"markers": cycle}, code="return this.data.y.map(i => markers[i]);"
        )
    }


lexis_ish_scatter = lexis_ish_figure.scatter(
    "x",
    "i",
    marker=cycle_mapper(markers),
    color=cycle_mapper(colors),
    source=scatter_source,
    legend_group="y",
)


# Add Lexis plot
lexis_figure = figure(
    title="Lexis plot",
    tools="tap,lasso_select," + default_tools,
    x_axis_label="Age at sample [months since 16]",
    y_axis_label="Date of sample",
)
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
slider.js_link("value", lexis_ish_scatter.glyph, "size")
# for line in lexis_lines:
#     slider.js_link("value", line.glyph, "size")

information_box = Div(
    text=f"""<h1>Welcome to the Simula Consulting interactive exploration demo!</h1>
    <br/><br/>

    <div style="border: 1px solid rgba(0,0,0,0.125); padding: 10px; border-radius: 5px; width: 50%;">
    Hei

    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed hendrerit nec elit quis consectetur. Cras in viverra quam. Donec vel dictum arcu. Suspendisse at mauris lobortis, efficitur magna sit amet, tristique nulla. Duis molestie mi vel consequat vestibulum. Pellentesque mollis aliquet lorem non varius. Sed placerat enim urna, in accumsan dui pharetra id. Donec ultricies convallis eleifend. Proin et nisl libero. Maecenas ultricies arcu tortor. Phasellus quis ante nisi. Aenean dictum est lorem, eu fermentum nibh mollis non.
    </div>
    <img src="data:image/png;base64, {SC_logo_base64}" style="position: absolute; top: -10px; right: 0;"/>
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
