"""Aligned statistical comparisons across saved forecast paths."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .metrics import qlike_loss, qlike_mean
from .tests import diebold_mariano


def _aligned_cell(
    forecasts: pd.DataFrame,
    models: Iterable[str],
    scheme: str,
    horizon: int,
) -> pd.DataFrame:
    """Return one aligned QLIKE-loss column per model."""
    columns = {}
    for model in models:
        cell = forecasts[
            forecasts["model"].eq(model)
            & forecasts["scheme"].eq(scheme)
            & forecasts["horizon"].eq(horizon)
        ].dropna(subset=["forecast_variance", "realized_variance"])
        if cell.empty:
            continue
        columns[model] = pd.Series(
            qlike_loss(cell["realized_variance"], cell["forecast_variance"]),
            index=pd.to_datetime(cell["forecast_target_date"]),
        )
    return pd.DataFrame(columns).dropna(how="any")


def dm_table(
    forecasts: pd.DataFrame,
    models: Iterable[str],
    baselines: Iterable[str] = ("B1.1", "B1.2"),
) -> pd.DataFrame:
    """Compute aligned Diebold-Mariano comparisons by scheme and horizon."""
    rows = []
    schemes = sorted(forecasts["scheme"].dropna().unique())
    horizons = sorted(forecasts["horizon"].dropna().unique())
    for scheme in schemes:
        for horizon in horizons:
            losses = _aligned_cell(forecasts, models, scheme, int(horizon))
            for baseline in baselines:
                if baseline not in losses:
                    continue
                for model in losses:
                    if model == baseline:
                        continue
                    mean_diff, statistic, p_value = diebold_mariano(
                        losses[model], losses[baseline], horizon=int(horizon)
                    )
                    rows.append(
                        {
                            "model": model,
                            "vs_baseline": baseline,
                            "scheme": scheme,
                            "horizon": int(horizon),
                            "mean_diff_QL": mean_diff,
                            "DM_stat": statistic,
                            "p_value": p_value,
                        }
                    )
    return pd.DataFrame(rows)


def model_confidence_set_table(
    forecasts: pd.DataFrame,
    models: Iterable[str],
    alpha: float = 0.10,
    reps: int = 5_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute the Hansen-Lunde-Nason Model Confidence Set for each cell."""
    try:
        from arch.bootstrap import MCS
    except ImportError as exc:
        raise ImportError("Model Confidence Set evaluation requires the 'arch' package.") from exc

    rows = []
    schemes = sorted(forecasts["scheme"].dropna().unique())
    horizons = sorted(forecasts["horizon"].dropna().unique())
    for scheme in schemes:
        for horizon in horizons:
            losses = _aligned_cell(forecasts, models, scheme, int(horizon))
            if losses.shape[0] < 30 or losses.shape[1] < 2:
                continue
            mcs = MCS(losses.to_numpy(), size=alpha, reps=reps, seed=seed, method="max")
            mcs.compute()
            included = [losses.columns[i] for i in mcs.included]
            excluded = [model for model in losses.columns if model not in included]
            rows.append(
                {
                    "scheme": scheme,
                    "horizon": int(horizon),
                    "n_obs": len(losses),
                    "in_MCS": ", ".join(included),
                    "excluded": ", ".join(excluded),
                }
            )
    return pd.DataFrame(rows)


def forecast_combination_table(
    forecasts: pd.DataFrame,
    model_a: str = "B1.2",
    model_b: str = "B2.2",
    weights: np.ndarray | None = None,
) -> pd.DataFrame:
    """Grid-search QLIKE-minimizing linear forecast-combination weights."""
    grid = np.linspace(0.0, 1.0, 21) if weights is None else np.asarray(weights)
    rows = []
    schemes = sorted(forecasts["scheme"].dropna().unique())
    horizons = sorted(forecasts["horizon"].dropna().unique())
    for scheme in schemes:
        for horizon in horizons:
            cell = forecasts[forecasts["scheme"].eq(scheme) & forecasts["horizon"].eq(horizon)]
            a = cell[cell["model"].eq(model_a)].set_index("forecast_target_date")
            b = cell[cell["model"].eq(model_b)].set_index("forecast_target_date")
            common = a.index.intersection(b.index)
            if common.empty:
                continue
            a, b = a.loc[common], b.loc[common]
            realized = a["realized_variance"].to_numpy()
            candidates = [
                qlike_mean(
                    realized,
                    (1.0 - weight) * a["forecast_variance"].to_numpy()
                    + weight * b["forecast_variance"].to_numpy(),
                )
                for weight in grid
            ]
            best = int(np.argmin(candidates))
            qlike_a = qlike_mean(realized, a["forecast_variance"])
            rows.append(
                {
                    "scheme": scheme,
                    "horizon": int(horizon),
                    "model_a": model_a,
                    "model_b": model_b,
                    "n_obs": len(common),
                    "QLIKE_a_alone": qlike_a,
                    "best_w": float(grid[best]),
                    "QLIKE_combined": float(candidates[best]),
                    "delta_QLIKE": float(candidates[best] - qlike_a),
                }
            )
    return pd.DataFrame(rows)
