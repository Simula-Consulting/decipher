import itertools
import json
from datetime import datetime

import pytest

from apps.landing_page.main import LandingPageFiltering


@pytest.fixture
def landing_page_filtering(tmp_path, monkeypatch):
    selected_pids_path = tmp_path / "selected_pids.json"
    monkeypatch.setattr(
        "apps.landing_page.main.settings.selected_pids_path", selected_pids_path
    )
    landing_page_filtering = LandingPageFiltering()
    return landing_page_filtering, selected_pids_path


def test_landing_page_filtering_save(landing_page_filtering):
    landing_page_filtering, selected_pids_path = landing_page_filtering

    landing_page_filtering.save_pids()
    assert selected_pids_path.is_file()
    with open(selected_pids_path, "r") as f:
        selected_pids = json.load(f)
    assert selected_pids == landing_page_filtering.pid_list


def test_landing_page_filtering_min_age_last_exam_slider_pids(landing_page_filtering):
    landing_page_filtering, selected_pids_path = landing_page_filtering

    value = 50
    landing_page_filtering.min_age_last_exam_slider.value = value
    pids = landing_page_filtering.min_age_last_exam_slider_pids()
    assert len(pids) == 35

    last_exam_after_value = landing_page_filtering.person_df["age_last_exam"] >= value
    not_selected_pids = ~landing_page_filtering.person_df.index.isin(pids)

    assert all(last_exam_after_value[pids])
    assert not any(last_exam_after_value[not_selected_pids])


def test_landing_page_filtering_min_n_screenings_slider_pids(landing_page_filtering):
    landing_page_filtering, selected_pids_path = landing_page_filtering

    value = 3
    landing_page_filtering.min_n_screenings_slider.value = value
    pids = landing_page_filtering.min_n_screenings_slider_pids()
    assert len(pids) == 39

    n_screenings_after_value = (
        landing_page_filtering.person_df["number_of_screenings"] >= value
    )
    not_selected_pids = ~landing_page_filtering.person_df.index.isin(pids)

    assert all(n_screenings_after_value[pids])
    assert not any(n_screenings_after_value[not_selected_pids])


def test_landing_page_filtering_age_slider_pids(landing_page_filtering):
    landing_page_filtering, selected_pids_path = landing_page_filtering

    # NB! the age slider name is misleading, it does expect calendar years, not ages.

    min_date, max_date = datetime(1970, 1, 1), datetime(1997, 12, 31)
    landing_page_filtering.age_slider.value = (min_date.year, max_date.year)
    pids = landing_page_filtering.age_slider_pids()
    assert len(pids) == 34

    year_in_range = landing_page_filtering.person_df["FOEDT"].between(
        min_date, max_date
    )
    not_selected_pids = ~landing_page_filtering.person_df.index.isin(pids)

    assert all(year_in_range[pids])
    assert not any(year_in_range[not_selected_pids])


def test_landing_page_filtering_checkbox_button_pids(landing_page_filtering):
    landing_page_filtering, selected_pids_path = landing_page_filtering

    # Maybe a bit lazy, but just check that activating more buttons gives fewer pids.
    # Iterator that gives ((), (1,), (1,2), ...)
    active_buttons_iter = itertools.accumulate(
        range(len(landing_page_filtering.checkbox_buttons.labels)),
        lambda previous, new: (*previous, new),
        initial=(),
    )
    filtered_pids = []
    for active_buttons in active_buttons_iter:
        landing_page_filtering.checkbox_buttons.active = list(active_buttons)
        filtered_pids.append(landing_page_filtering.checkbox_button_pids())
    assert (a <= b for a, b in itertools.pairwise(filtered_pids))
