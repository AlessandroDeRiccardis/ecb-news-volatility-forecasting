"""Generate core descriptive tables and figures from available artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_core_artifacts(df: pd.DataFrame, tables: Path, figures: Path) -> None:
    """Write descriptive statistics and two data-overview figures."""
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    rows = []
    for period in ("insample", "outsample"):
        sample = df[df["period"] == period]
        rows.append(
            {
                "period": period,
                "n": len(sample),
                "return_mean": sample["log_return"].mean(),
                "return_std": sample["log_return"].std(),
                "return_skew": sample["log_return"].skew(),
                "return_kurtosis": sample["log_return"].kurt(),
                "P_t_mean": sample["P_t"].mean(),
                "N_t_mean": sample["N_t"].mean(),
            }
        )
    pd.DataFrame(rows).to_csv(tables / "descriptive_statistics.csv", index=False)

    plotted = df[df["period"] != "presample"].copy()
    plotted["vol_20d"] = plotted["log_return"].rolling(20).std() * np.sqrt(252) * 100
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(plotted["date"], plotted["log_return"] * 100, linewidth=0.5)
    axes[0].set_ylabel("Return (%)")
    axes[1].plot(plotted["date"], plotted["vol_20d"], linewidth=0.8)
    axes[1].set_ylabel("20-day ann. vol (%)")
    axes[1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(figures / "returns_and_realized_volatility.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(plotted["date"], plotted["P_t"], label="Dovish P_t", linewidth=0.8)
    ax.plot(plotted["date"], plotted["N_t"], label="Hawkish N_t", linewidth=0.8)
    ax.plot(plotted["date"], plotted["S_t"], label="Net S_t", linewidth=0.6, alpha=0.6)
    ax.axhline(0, color="black", linewidth=0.4)
    ax.legend()
    ax.set_xlabel("Date")
    ax.set_ylabel("Rescaled stance")
    fig.tight_layout()
    fig.savefig(figures / "daily_stance_series.png", dpi=180)
    plt.close(fig)
