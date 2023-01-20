from datetime import datetime
from typing import Callable

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames, range_indexes, series

from matfact.data_generation.processing import ScreeningDataProcessingPipeline
from matfact.settings import settings


@st.composite
def generate_mock_screening_data(draw: Callable) -> pd.DataFrame:
    pid_column = column(
        name=settings.processing.pid, elements=st.integers(min_value=1, max_value=100)
    )

    cyt_col, hist_col = settings.processing.cyt, settings.processing.hist
    cyt_results = st.sampled_from(
        list(settings.processing.risk_maps[cyt_col.name].keys()) + 5 * [np.nan]
    )
    cyt_diag_column = column(name=settings.processing.cyt.name, elements=cyt_results)

    mock_screening_data = draw(
        data_frames(
            columns=[pid_column, cyt_diag_column],
            index=range_indexes(min_size=1, max_size=150),
        )
    )
    size = len(mock_screening_data)
    dates = np.array(
        [
            datetime.strftime(x, format=settings.processing.dateformat)
            for x in draw(
                st.lists(
                    st.datetimes(
                        min_value=datetime(1950, 1, 1), max_value=datetime(2020, 1, 1)
                    ),
                    min_size=size,
                    max_size=size,
                )
            )
        ]
    )
    cyt_idx = mock_screening_data[cyt_col.name].notna().values
    mock_screening_data.loc[cyt_idx, cyt_col.date] = dates[cyt_idx]

    hist_results = np.array(
        draw(
            st.lists(
                st.sampled_from(
                    list(settings.processing.risk_maps[hist_col.name].keys())
                ),
                min_size=size,
                max_size=size,
            )
        )
    )
    mock_screening_data.loc[~cyt_idx, hist_col.name] = hist_results[~cyt_idx]
    mock_screening_data.loc[~cyt_idx, hist_col.date] = dates[~cyt_idx]

    return mock_screening_data


@st.composite
def generate_mock_dob_data(draw: Callable) -> pd.DataFrame:
    bday_strat = st.datetimes(
        min_value=datetime(1900, 1, 1), max_value=datetime(1949, 1, 1)
    )
    birthdate_column = column(name=settings.processing.dob.date, elements=bday_strat)
    status_column = column(
        name=settings.processing.dob.name, elements=st.sampled_from(["B", "D", np.nan])
    )

    df = draw(
        data_frames(
            columns=[status_column, birthdate_column],
            index=range_indexes(min_size=1, max_size=150),
        )
    )
    df[settings.processing.dob.date] = df[settings.processing.dob.date].apply(
        lambda x: datetime.strftime(x, format=settings.processing.dateformat)
    )
    df[settings.processing.pid] = list(range(1, len(df) + 1))
    return df


@given(
    screening_data=generate_mock_screening_data(),
    dob_data=generate_mock_dob_data(),
    n_females=st.one_of(st.none(), st.integers(min_value=1, max_value=100)),
)
def test_prepare_data(screening_data, dob_data, n_females) -> None:
    processor = ScreeningDataProcessingPipeline(
        screening_data, dob_data, n_females=n_females
    )
    prepared_data = processor.prepare_data()
