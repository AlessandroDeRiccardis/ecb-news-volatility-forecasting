"""Schema checks for the processed modeling dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "date",
    "period",
    "log_return",
    "sq_return",
    "abs_return",
    "sq_return_5d",
    "P_t",
    "N_t",
    "S_t",
}
VALID_PERIODS = {"presample", "insample", "outsample"}


class MasterDataError(ValueError):
    """Raised when the processed master dataset violates its contract."""


def validate_master_dataset(df: pd.DataFrame) -> None:
    """Validate columns, ordering, period labels, and key numerical invariants."""
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise MasterDataError(f"Missing required columns: {missing}")
    if df.empty:
        raise MasterDataError("Master dataset is empty.")
    dates = pd.to_datetime(df["date"], errors="coerce")
    if dates.isna().any():
        raise MasterDataError("Column 'date' contains invalid values.")
    if not dates.is_monotonic_increasing or dates.duplicated().any():
        raise MasterDataError("Dates must be unique and increasing.")
    unknown_periods = set(df["period"].dropna().unique()) - VALID_PERIODS
    if unknown_periods:
        raise MasterDataError(f"Unknown period labels: {sorted(unknown_periods)}")
    if not np.isfinite(df["log_return"]).all():
        raise MasterDataError("log_return must be finite.")
    if (df["sq_return"] < 0).any() or (df["abs_return"] < 0).any():
        raise MasterDataError("Realized volatility proxies must be non-negative.")
    if ((df["N_t"].dropna() > 0) | (df["P_t"].dropna() < 0)).any():
        raise MasterDataError("Stance support violated: expected P_t >= 0 and N_t <= 0.")


def load_master_dataset(path: str | Path) -> pd.DataFrame:
    """Load and validate the processed master dataset."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing {dataset_path}. See data/README.md for required inputs.")
    df = pd.read_csv(dataset_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    validate_master_dataset(df)
    return df
