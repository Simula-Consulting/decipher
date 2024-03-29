import json

from bokeh.io import curdoc
from bokeh.models import Model
from bokeh.models.callbacks import CustomJS
from bokeh.models.widgets import Button, CheckboxButtonGroup, Div, RangeSlider, Slider
from decipher.data import DataManager
from loguru import logger

from bokeh_demo.data_ingestion import add_hpv_detailed_information
from bokeh_demo.settings import settings


class LandingPageFiltering:
    def __init__(self):
        self.column_names = settings.feature_column_names

        self.data_manager = self._load_data_manager()
        self.person_df = self.data_manager.person_df
        self.exams_df = add_hpv_detailed_information(
            self.data_manager.exams_df, self.data_manager.hpv_df
        )
        self._add_HR_screening_indicator()

        self.pid_list = self.person_df.index.to_list()
        """PID selection from currently active filters"""

        self.birthyear_slider = self._init_birthyear_slider()
        self.checkbox_buttons = self._init_checkbox_buttons()
        self.min_age_last_exam_slider = self._init_min_age_last_exam_slider()
        self.min_n_screenings_slider = self._init_min_n_screenings_slider()

        self.save_button = self._init_save_button()
        self.person_counter = Div(text=f"Number of people: {self._n_people()}")
        self.go_button = self._init_go_button()

    def get_roots(self) -> list[Model]:
        return [
            self.birthyear_slider,
            self.min_age_last_exam_slider,
            self.min_n_screenings_slider,
            self.checkbox_buttons,
            self.person_counter,
            self.save_button,
            self.go_button,
        ]

    def reset_pid_list(self) -> None:
        self.pid_list = self.person_df.index.to_list()

    def _n_people(self) -> int:
        return len(self.pid_list)

    def _add_HR_screening_indicator(self) -> None:
        for exam_type, df in self.exams_df.groupby(self.column_names.exam_type):
            if exam_type == "HPV":
                continue
            column_name = f"high_risk_{exam_type.lower()}"
            self.person_df.loc[
                df[
                    df[self.column_names.risk] > 2
                ][  # 3 and above are considered high risk
                    self.column_names.PID
                ].unique(),
                column_name,
            ] = True
            self.person_df[column_name].fillna(False, inplace=True)

    def apply_filters_and_save(self) -> None:
        self.reset_pid_list()
        birthyear_pids = self.birthyear_slider_pids()
        button_pids = self.checkbox_button_pids()
        min_age_last_exam_pids = self.min_age_last_exam_slider_pids()
        min_n_screenings_pids = self.min_n_screenings_slider_pids()

        self.pid_list = list(
            map(
                int,
                list(
                    set(birthyear_pids)
                    & set(button_pids)
                    & set(min_age_last_exam_pids)
                    & set(min_n_screenings_pids)
                ),
            )
        )
        self.person_counter.text = f"Number of people: {self._n_people()}"
        self.save_pids()

    def _init_go_button(self) -> Button:
        go_button = Button(label="Go")
        go_button.js_on_click(CustomJS(code="window.location.href = '/pilot'"))
        return go_button

    def _init_save_button(self) -> Button:
        save_button = Button(label="Save")
        save_button.on_click(self.apply_filters_and_save)
        return save_button

    def _init_birthyear_slider(self) -> RangeSlider:
        birthyears = self.person_df[self.column_names.birthdate].dt.year
        start, end = birthyears.min(), birthyears.max()
        return RangeSlider(
            name="birthyear_slider",
            start=start,
            end=end,
            value=(start, end),
            step=1,
            title="Birth Year",
        )

    def birthyear_slider_pids(self) -> list[int]:
        year_min, year_max = self.birthyear_slider.value
        return list(
            self.person_df[
                self.person_df[self.column_names.birthdate].dt.year.between(
                    year_min, year_max
                )
            ].index
        )

    def _pids_from_button(self, button_name: str) -> list[int]:
        """Function to get a list of pids for the people that have the selected attribute."""

        def _get_hpv_pids(hpv_type: str) -> list[int]:
            """Function to get the pids that have the selected HPV type"""
            hpv_filter = (self.exams_df[self.column_names.exam_type] == "HPV") & (
                self.exams_df[self.column_names.exam_diagnosis] == "positiv"
            )
            hpv_results = self.exams_df[hpv_filter][
                self.column_names.exam_details
            ].str.split(",")

            hpv_inds = hpv_results[hpv_results.apply(lambda x: hpv_type in x)].index
            hpv_pids = self.exams_df.loc[hpv_inds, self.column_names.PID].tolist()
            return hpv_pids

        match button_name:
            case "HPV+":
                return list(
                    self.person_df[
                        self.person_df[self.column_names.hpv_pos_count] > 0
                    ].index
                )
            case "HR Result":
                return list(
                    self.person_df[self.person_df[self.column_names.risk_max] > 2].index
                )
            case "HR Cytology":
                return list(
                    self.person_df[self.person_df[self.column_names.hr_cytology]].index
                )
            case "HR Histology":
                return list(
                    self.person_df[self.person_df[self.column_names.hr_histology]].index
                )
            case "HPV16+":
                return _get_hpv_pids("16")
            case "HPV18+":
                return _get_hpv_pids("18")
            case _:
                return []

    def checkbox_button_pids(self) -> list[int]:
        """Function to get the pids that have the selected attributes"""
        active_button_names: list[str] = [
            self.button_positions[i] for i in self.checkbox_buttons.active
        ]

        button_pids = set(self.pid_list)
        for button_name in active_button_names:
            button_pids = button_pids.intersection(
                set(self._pids_from_button(button_name))
            )
        return list(button_pids)

    def _init_checkbox_buttons(self) -> CheckboxButtonGroup:
        labels = [
            "HPV+",
            "HR Result",
            "HR Cytology",
            "HR Histology",
            "HPV16+",
            "HPV18+",
        ]
        self.button_positions = {i: label for i, label in enumerate(labels)}
        return CheckboxButtonGroup(
            name="checkbox_buttons",
            labels=labels,
        )

    def min_age_last_exam_slider_pids(self) -> list[int]:
        """Function to get the pids that have the selected attributes"""
        min_age_last_exam = self.min_age_last_exam_slider.value
        return self.person_df[
            self.person_df[self.column_names.age_last_exam] >= min_age_last_exam
        ].index.to_list()

    def _init_min_age_last_exam_slider(self) -> Slider:
        return Slider(
            name="min_age_last_exam",
            start=0,
            end=100,
            value=0,
            step=1,
            title="Minimum age at last exam",
        )

    def min_n_screenings_slider_pids(self) -> list[int]:
        """Function to get the pids that have the selected attributes"""
        min_n_screenings = self.min_n_screenings_slider.value
        return self.person_df[
            self.person_df[self.column_names.n_screenings] >= min_n_screenings
        ].index.to_list()

    def _init_min_n_screenings_slider(self) -> Slider:
        return Slider(
            name="min_n_screenings",
            start=0,
            end=10,
            value=0,
            step=1,
            title="Minimum number of screenings",
        )

    def _load_data_manager(self) -> DataManager:
        """Function to load the DataFrame from the parquet file.
        If the parquet file is not found, it will fall back to loading the csv file.
        """
        try:
            data_manager = DataManager.from_parquet(settings.data_paths.base_path)
        except (FileNotFoundError, ImportError, ValueError) as e:
            logger.exception(e)
            logger.warning(
                "Falling back to .csv loading. This will affect performance."
            )
            data_manager = DataManager.read_from_csv(
                settings.data_paths.screening_data_path,
                settings.data_paths.dob_data_path,
                read_hpv=True,
            )
        return data_manager

    def save_pids(self):
        """Save pid_list as json in `settings.selected_pids_path`."""
        with open(settings.selected_pids_path, "w") as f:
            json.dump(self.pid_list, f)


landing_page = LandingPageFiltering()
for root_element in landing_page.get_roots():
    curdoc().add_root(root_element)
