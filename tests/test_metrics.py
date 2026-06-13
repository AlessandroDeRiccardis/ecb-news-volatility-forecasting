from __future__ import annotations

import numpy as np
import pandas as pd

from ecb_vol_forecasting.evaluation import (
    diebold_mariano,
    forecast_combination_table,
    qlike_loss,
    qlike_mean,
    rmse,
)


def test_qlike_is_finite_for_zero_forecast() -> None:
    loss = qlike_loss([0.0, 0.01], [0.0, 0.02])
    assert np.isfinite(loss).all()


def test_qlike_and_rmse_prefer_correct_forecast() -> None:
    realized = np.array([0.01, 0.02, 0.03])
    assert qlike_mean(realized, realized) < qlike_mean(realized, realized * 2)
    assert rmse(realized, realized) == 0.0


def test_dm_returns_finite_values() -> None:
    mean_diff, statistic, p_value = diebold_mariano(
        np.linspace(1.0, 2.0, 20),
        np.linspace(1.1, 2.1, 20),
        horizon=5,
    )
    assert np.isfinite([mean_diff, statistic, p_value]).all()
    assert 0.0 <= p_value <= 1.0


def test_forecast_combination_selects_better_model() -> None:
    rows = []
    for model, forecast in (("B1.2", 1.0), ("B2.2", 2.0)):
        for date in pd.date_range("2020-01-01", periods=10):
            rows.append(
                {
                    "model": model,
                    "scheme": "RW",
                    "horizon": 1,
                    "forecast_target_date": date,
                    "forecast_variance": forecast,
                    "realized_variance": 1.0,
                }
            )
    result = forecast_combination_table(pd.DataFrame(rows))
    assert result.loc[0, "best_w"] == 0.0
