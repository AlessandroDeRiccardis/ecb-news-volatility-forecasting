"""Tabular evaluation of saved OOS forecasts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .metrics import qlike_mean, rmse


def load_forecasts(directory: str | Path) -> pd.DataFrame:
    """Load all per-model forecast files in a directory."""
    files = sorted(Path(directory).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No forecast CSVs found in {directory}")
    return pd.concat(
        [pd.read_csv(path, parse_dates=["origin_date", "forecast_target_date"]) for path in files],
        ignore_index=True,
    )


def accuracy_table(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Compute QLIKE and RMSE by model, scheme, and horizon."""
    rows = []
    for keys, group in forecasts.groupby(["model", "scheme", "horizon"]):
        model, scheme, horizon = keys
        clean = group.dropna(subset=["forecast_variance", "realized_variance"])
        rows.append(
            {
                "model": model,
                "scheme": scheme,
                "horizon": int(horizon),
                "n_obs": len(clean),
                "QLIKE": qlike_mean(clean["realized_variance"], clean["forecast_variance"]),
                "RMSE": rmse(clean["realized_variance"], clean["forecast_variance"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["scheme", "horizon", "QLIKE"])
