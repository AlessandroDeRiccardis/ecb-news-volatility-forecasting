from __future__ import annotations

import pandas as pd
import pytest

from ecb_vol_forecasting.data import (
    MasterDataError,
    build_forecast_origins,
    training_window,
    validate_master_dataset,
)


def test_master_schema_accepts_valid_data(synthetic_master: pd.DataFrame) -> None:
    validate_master_dataset(synthetic_master)


def test_master_schema_rejects_duplicate_dates(synthetic_master: pd.DataFrame) -> None:
    synthetic_master.loc[1, "date"] = synthetic_master.loc[0, "date"]
    with pytest.raises(MasterDataError):
        validate_master_dataset(synthetic_master)


def test_training_windows_have_no_lookahead(synthetic_master: pd.DataFrame) -> None:
    origins = build_forecast_origins(synthetic_master, step=5)
    for origin in origins:
        for scheme in ("RW", "IW"):
            train = training_window(synthetic_master, origin, scheme)
            assert train["date"].max() == synthetic_master.iloc[origin]["date"]
            assert (train["date"] <= synthetic_master.iloc[origin]["date"]).all()


def test_rolling_window_never_exceeds_in_sample_size(synthetic_master: pd.DataFrame) -> None:
    in_sample_size = synthetic_master["period"].eq("insample").sum()
    for origin in build_forecast_origins(synthetic_master):
        assert len(training_window(synthetic_master, origin, "RW")) <= in_sample_size
