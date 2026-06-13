"""Lightweight Bernoth-style event residualization."""

from __future__ import annotations

import numpy as np
import pandas as pd


def event_residual_surprises(
    frame: pd.DataFrame,
    target: str,
    controls: list[str],
    event_mask: pd.Series,
    insample_mask: pd.Series,
) -> pd.Series:
    """Fit OLS on in-sample event days and return full-sample event residuals."""
    columns = [target, *controls]
    estimation = frame.loc[event_mask & insample_mask, columns].dropna()
    if len(estimation) <= len(controls) + 1:
        raise ValueError("Insufficient event observations for residualization.")
    x = np.column_stack([np.ones(len(estimation)), estimation[controls].to_numpy()])
    beta, *_ = np.linalg.lstsq(x, estimation[target].to_numpy(), rcond=None)
    residuals = pd.Series(np.nan, index=frame.index, dtype=float)
    events = frame.loc[event_mask, columns].dropna()
    x_full = np.column_stack([np.ones(len(events)), events[controls].to_numpy()])
    residuals.loc[events.index] = events[target].to_numpy() - x_full @ beta
    return residuals
