from bokeh.io import curdoc
from bokeh.models import TextInput, Button, CustomJS
import pickle
from bokeh_demo.settings import settings

text = TextInput(title="Text input", value="input your text here")


def save_text(attr, old, new):
    with open(settings.selected_pids_path, "wb") as f:
        pickle.dump(text.value, f)


go_button = Button(label="Go")
go_button_js_callback = CustomJS(
    code="""
window.location.href = "/pilot"
"""
)
go_button.js_on_click(go_button_js_callback)

# currently saving file on every change
text.on_change("value", save_text)

for root_element in (text, go_button):
    curdoc().add_root(root_element)
