import json
import warnings

from bokeh.io import curdoc
from bokeh.models import Model
from bokeh.models.callbacks import CustomJS
from bokeh.models.widgets import Button, CheckboxButtonGroup, Div, RangeSlider, Slider
from decipher.data import DataManager

from bokeh_demo.settings import settings


class LandingPageFiltering:
    def __init__(self):
        self.column_names = settings.feature_column_names

        self.data_manager = self._load_data_manager()
        self.person_df = self.data_manager.person_df
        self.pid_list = self.person_df.index.to_list()

        self.age_slider = self._init_age_slider()
        self.checkbox_buttons = self._init_checkbox_buttons()
        self.min_age_last_exam_slider = self._init_min_age_last_exam_slider()

        self.save_button = self._init_save_button()
        self.person_counter = Div(text=f"Number of people: {self._n_people()}")
        self.go_button = self._init_go_button()

    def get_roots(self) -> list[Model]:
        return [
            self.age_slider,
            self.min_age_last_exam_slider,
            self.checkbox_buttons,
            self.person_counter,
            self.save_button,
            self.go_button,
        ]

    def reset_pid_list(self) -> None:
        self.pid_list = self.person_df.index.to_list()

    def _n_people(self) -> int:
        return len(self.pid_list)

    def apply_filters_and_save(self) -> None:
        self.reset_pid_list()
        age_pids = self.age_slider_pids()
        button_pids = self.checkbox_button_pids()
        min_age_last_exam_pids = self.min_age_last_exam_slider_pids()

        self.pid_list = list(
            map(
                int,
                list(set(age_pids) & set(button_pids) & set(min_age_last_exam_pids)),
            )
        )
        self.person_counter.text = f"Number of people: {self._n_people()}"
        self.save_pids()

    def _init_go_button(self) -> Button:
        go_button = Button(label="Go")
        go_button_js_callback = CustomJS(code="window.location.href = '/pilot'")
        go_button.js_on_click(go_button_js_callback)
        return go_button

    def _init_save_button(self) -> Button:
        save_button = Button(label="Save")
        save_button.on_click(self.apply_filters_and_save)
        return save_button

    def _init_age_slider(self) -> RangeSlider:
        birthyears = self.person_df[self.column_names.birthdate].dt.year
        start, end = birthyears.min(), birthyears.max()
        age_slider = RangeSlider(
            name="age_slider",
            start=start,
            end=end,
            value=(start, end),
            step=1,
            title="Birth Year",
        )
        return age_slider

    def age_slider_pids(self) -> list[int]:
        age_min, age_max = self.age_slider.value

        age_pids = list(
            self.person_df[
                self.person_df[self.column_names.birthdate].dt.year.between(
                    age_min, age_max
                )
            ].index
        )
        return age_pids

    def checkbox_button_pids(self) -> list[int]:
        """Function to get the pids that have the selected attributes"""
        buttons_active: list[int] = self.checkbox_buttons.active
        button_pids = set(self.pid_list)
        if 0 in buttons_active:
            # Has HPV+
            button_pids = button_pids.intersection(
                set(
                    self.person_df[
                        self.person_df[self.column_names.hpv_pos_count] > 0
                    ].index
                )
            )
        if 1 in buttons_active:
            # Has High Risk Result
            button_pids = button_pids.intersection(
                self.person_df[self.person_df[self.column_names.risk_max] > 2].index
            )
        return list(button_pids)

    def _init_checkbox_buttons(self) -> CheckboxButtonGroup:
        labels = ["Has HPV+", "Has High Risk Result"]

        buttons = CheckboxButtonGroup(
            name="checkbox_buttons",
            labels=labels,
        )
        return buttons

    def min_age_last_exam_slider_pids(self) -> list[int]:
        """Function to get the pids that have the selected attributes"""
        min_age_last_exam = self.min_age_last_exam_slider.value
        return self.person_df[
            self.person_df[self.column_names.age_last_exam] >= min_age_last_exam
        ].index.to_list()

    def _init_min_age_last_exam_slider(self) -> Slider:
        min_age_last_exam_slider = Slider(
            name="min_age_last_exam",
            start=0,
            end=100,
            value=0,
            step=1,
            title="Minimum age at last exam",
        )
        return min_age_last_exam_slider

    def _load_data_manager(self) -> DataManager:
        """Function to load the dataframe from the parquet file.
        If the parquet file is not found, it will fall back to loading the csv file.
        """
        try:
            data_manager = DataManager.from_parquet(settings.data_paths.base_path)
        except (FileNotFoundError, ImportError, ValueError) as e:
            warnings.warn(str(e))
            print("Falling back to .csv loading. This will affect performance.")
            data_manager = DataManager.read_from_csv(
                settings.data_paths.screening_data_path,
                settings.data_paths.dob_data_path,
            )
        # Add column for age in years
        return data_manager

    def save_pids(self):
        with open(settings.selected_pids_path, "w") as f:
            json.dump(self.pid_list, f)


landing_page = LandingPageFiltering()
for root_element in landing_page.get_roots():
    curdoc().add_root(root_element)
