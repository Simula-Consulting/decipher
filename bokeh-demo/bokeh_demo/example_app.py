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

dataset_path = pathlib.Path('..') / "data/dataset1"

dataset = Dataset.from_file(dataset_path)
data = pd.DataFrame(dataset.X.astype(int), columns=(f"step_{t:02}" for t in range(len(dataset.X[0]))))

# Fake deltas
deltas = [random() * 2 - 1 for _ in range(dataset.X.shape[0])]

# create a plot and style its properties
p = figure(tools="tap,box_select")
# p.border_fill_color = 'black'
# p.background_fill_color = 'black'
# p.outline_line_color = None
# p.grid.grid_line_color = None

# add a text renderer to the plot (no data yet)
source = ColumnDataSource({"x": list(range(dataset.X.shape[0])), "y": deltas})
r = p.circle(source=source)
# text_font_size="26px", text_baseline="middle", text_align="center", )

i = 0


xs = list(itertools.repeat(list(range(dataset.X.shape[1])), dataset.X.shape[0]))
ys = dataset.X.tolist()
line_source = ColumnDataSource({"xs": xs, "ys": ys})
line_view = CDSView(source=line_source, filters=[])

def print_attr(attr, old, new):
    print(f"{attr} changed from {old} to {new}")
    if attr == "indices":
        source.selected.indices = new
        line_source.selected.indices = new
        line_view.filters = [IndexFilter(new)] if new else []

# create a callback that adds a number in a random location
def callback():
    global i

    # BEST PRACTICE --- update .data in one step with a new dict
    new_data = dict()
    new_data['x'] = source.data['x'] + [random()*70 + 15]
    new_data['y'] = source.data['y'] + [random()*70 + 15]
    new_data['text_color'] = source.data['text_color'] + [RdYlBu3[i%3]]
    new_data['text'] = source.data['text'] + [str(i)]
    source.data = new_data

    i = i + 1

# add a button widget and configure with the call back
button = Button(label="Press Me")
button.on_click(callback)

selected_indices = []
def show_selected(attr, old, new):
    print(source.selected)
    print(source.selected.indices)
    print(attr, old, new)
    selected_indices = new

source.selected.on_change("indices", print_attr)
source.selected.js_on_change("indices", CustomJS(code="""
console.log('hei');
"""))

logs = figure(tools="tap,lasso_select")
lines=logs.multi_line(xs="xs", ys="ys", source=line_source, view=line_view)


line_source.selected.on_change("indices", print_attr)
line_source.selected.on_change("multiline_indices", print_attr)


# put the button and plot in a layout and add to the document
curdoc().add_root(
    row(
        column(button, p),
        logs,
    )
)