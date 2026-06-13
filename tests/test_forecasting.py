from __future__ import annotations

import numpy as np
import pandas as pd

from ecb_vol_forecasting.models.forecasting import run_forecasts


class FakeVarianceModel:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame

    def fit(self):
        return self

    def forecast_variance(self, h: int = 1):
        return np.full(h, max(float(self.frame["sq_return"].mean()), 1e-8))


def fake_factory(_name: str, frame: pd.DataFrame, _distribution: str) -> FakeVarianceModel:
    return FakeVarianceModel(frame)


def test_forecast_shapes_and_positive_variances(synthetic_master: pd.DataFrame) -> None:
    forecasts = run_forecasts(
        synthetic_master,
        model_name="B1.1",
        scheme="RW",
        horizon=5,
        step=5,
        model_factory=fake_factory,
    )
    assert not forecasts.empty
    assert set(forecasts.columns) == {
        "origin_date",
        "forecast_target_date",
        "model",
        "scheme",
        "horizon",
        "forecast_variance",
        "realized_variance",
        "abs_return_target",
        "n_train",
    }
    assert (forecasts["forecast_variance"] > 0).all()
