"""Forecast-origin and no-look-ahead training-window helpers."""

from __future__ import annotations

import pandas as pd


def build_forecast_origins(df: pd.DataFrame, step: int = 5) -> list[int]:
    """Return weekly-style origins from the last in-sample row through OOS."""
    if step < 1:
        raise ValueError("step must be positive.")
    insample = df.index[df["period"] == "insample"].tolist()
    outsample = df.index[df["period"] == "outsample"].tolist()
    if not insample or not outsample:
        raise ValueError("Both insample and outsample rows are required.")
    first_origin = insample[-1]
    last_origin = min(outsample[-1], len(df) - 2)
    return list(range(first_origin, last_origin + 1, step))


def training_window(df: pd.DataFrame, origin: int, scheme: str) -> pd.DataFrame:
    """Return data available at an origin under rolling or increasing windows."""
    if origin < 0 or origin >= len(df):
        raise IndexError("origin is outside the dataset.")
    insample_idx = df.index[df["period"] == "insample"].tolist()
    if not insample_idx:
        raise ValueError("No in-sample rows found.")
    if scheme == "RW":
        start = max(insample_idx[0], origin - len(insample_idx) + 1)
    elif scheme == "IW":
        start = insample_idx[0]
    else:
        raise ValueError("scheme must be 'RW' or 'IW'.")
    return df.iloc[start : origin + 1].copy()
