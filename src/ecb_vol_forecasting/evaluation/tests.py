"""Statistical forecast-comparison tests."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def diebold_mariano(loss_a, loss_b, horizon: int = 1) -> tuple[float, float, float]:
    """Two-sided DM test with Newey-West bandwidth h-1."""
    differential = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    differential = differential[np.isfinite(differential)]
    if len(differential) < 5:
        return np.nan, np.nan, np.nan
    mean_diff = float(differential.mean())
    long_run_variance = float(np.var(differential, ddof=1))
    for lag in range(1, max(horizon - 1, 0) + 1):
        covariance = float(
            np.mean((differential[lag:] - mean_diff) * (differential[:-lag] - mean_diff))
        )
        weight = 1.0 - lag / horizon
        long_run_variance += 2.0 * weight * covariance
    long_run_variance = max(long_run_variance, 1e-12)
    statistic = mean_diff / np.sqrt(long_run_variance / len(differential))
    p_value = 2.0 * (1.0 - norm.cdf(abs(statistic)))
    return mean_diff, float(statistic), float(p_value)
