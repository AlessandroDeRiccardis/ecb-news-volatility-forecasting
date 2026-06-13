"""No-look-ahead rolling and increasing-window forecasts."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from ecb_vol_forecasting.data.splits import build_forecast_origins, training_window

from .factory import clean_training_data, is_news_augmented, make_model

ModelFactory = Callable[[str, pd.DataFrame, str], object]


def run_forecasts(
    df: pd.DataFrame,
    model_name: str,
    scheme: str,
    horizon: int,
    distribution: str = "studentst",
    step: int = 5,
    max_origins: int | None = None,
    model_factory: ModelFactory = make_model,
) -> pd.DataFrame:
    """Estimate at each origin and return aligned variance forecasts."""
    if horizon < 1:
        raise ValueError("horizon must be positive.")
    origins = build_forecast_origins(df, step=step)
    if max_origins is not None:
        origins = origins[:max_origins]

    rows: list[dict[str, object]] = []
    warm_theta: np.ndarray | None = None
    for origin in origins:
        train = clean_training_data(training_window(df, origin, scheme), model_name)
        if len(train) < 20:
            continue
        model = model_factory(model_name, train, distribution)
        if is_news_augmented(model_name):
            kwargs = {"n_restarts": 1}
            if warm_theta is not None and hasattr(model, "_theta_init"):
                if len(warm_theta) == len(model._theta_init()):
                    kwargs = {"n_restarts": 0, "warm_start": {"theta": warm_theta}}
            model.fit(**kwargs)
            warm_theta = model._theta_opt.copy()
        else:
            model.fit()
        path = np.asarray(model.forecast_variance(h=horizon), dtype=float)
        forecast = float(path[0] if horizon == 1 else path.sum())
        target = origin + 1
        if target >= len(df):
            continue
        realized = (
            float(df.iloc[target]["sq_return"])
            if horizon == 1
            else float(df.iloc[target]["sq_return_5d"])
        )
        abs_target = float(df.iloc[target : target + horizon]["abs_return"].sum())
        rows.append(
            {
                "origin_date": df.iloc[origin]["date"],
                "forecast_target_date": df.iloc[target]["date"],
                "model": model_name,
                "scheme": scheme,
                "horizon": horizon,
                "forecast_variance": forecast,
                "realized_variance": realized,
                "abs_return_target": abs_target,
                "n_train": len(train),
            }
        )
    return pd.DataFrame(rows)
