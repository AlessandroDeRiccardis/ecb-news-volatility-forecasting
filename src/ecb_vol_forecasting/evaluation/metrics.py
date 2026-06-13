"""Numerically stable forecast loss functions."""

from __future__ import annotations

import numpy as np


def qlike_loss(realized, forecast, epsilon: float = 1e-12) -> np.ndarray:
    """Return observation-level Patton QLIKE losses."""
    r = np.asarray(realized, dtype=float)
    f = np.asarray(forecast, dtype=float)
    valid = np.isfinite(r) & np.isfinite(f) & (r >= 0)
    out = np.full(np.broadcast_shapes(r.shape, f.shape), np.nan, dtype=float)
    safe_f = np.maximum(f, epsilon)
    out[valid] = np.log(safe_f[valid]) + r[valid] / safe_f[valid]
    return out


def qlike_mean(realized, forecast, epsilon: float = 1e-12) -> float:
    """Return mean QLIKE over finite observations."""
    return float(np.nanmean(qlike_loss(realized, forecast, epsilon=epsilon)))


def rmse(realized, forecast) -> float:
    """Return root mean squared forecast error."""
    r = np.asarray(realized, dtype=float)
    f = np.asarray(forecast, dtype=float)
    return float(np.sqrt(np.nanmean((r - f) ** 2)))
