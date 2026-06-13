"""Model registry for the empirical suite."""

from __future__ import annotations

import pandas as pd

from .garch import EGARCH, GARCH11, GJRGARCH, NAGarchAsym, NAGarchNet

MAIN_MODELS = ("B1.1", "B1.2", "B1.3", "B2.1", "B2.2")
SOURCE_ROBUSTNESS_MODELS = ("B5.3", "B5.4", "B5.5")
MODEL_LABELS = {
    "B1.1": "GARCH(1,1)",
    "B1.2": "GJR-GARCH(1,1)",
    "B1.3": "EGARCH(1,1)",
    "B2.1": "NA-GARCH-net (paper: NA1.1)",
    "B2.2": "NA-GARCH-asym (paper: NA1.2)",
    "B2.3": "NA-GARCH-asym MPS surprise",
    "B5.3": "NA-GARCH-asym MPS only",
    "B5.4": "NA-GARCH-asym speeches only",
    "B5.5": "NA-GARCH-asym bulletins only",
    "B5.6": "NA-GARCH-asym threshold 0.00",
    "B5.7": "NA-GARCH-asym threshold 0.80",
}
MODEL_COLUMNS = {
    "B1.1": ("log_return",),
    "B1.2": ("log_return",),
    "B1.3": ("log_return",),
    "B2.1": ("log_return", "S_t"),
    "B2.2": ("log_return", "P_t", "N_t"),
    "B2.3": ("log_return", "P_mps_surprise", "N_mps_surprise"),
    "B5.3": ("log_return", "P_t_mps", "N_t_mps"),
    "B5.4": ("log_return", "P_t_speech", "N_t_speech"),
    "B5.5": ("log_return", "P_t_bulletins", "N_t_bulletins"),
    "B5.6": ("log_return", "P_t_t000", "N_t_t000"),
    "B5.7": ("log_return", "P_t_t080", "N_t_t080"),
}


def is_news_augmented(model_name: str) -> bool:
    """Return whether a model uses an exogenous stance series."""
    return model_name.startswith(("B2.", "B5."))


def clean_training_data(frame: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Drop rows missing the exact columns required by a model."""
    try:
        columns = MODEL_COLUMNS[model_name]
    except KeyError as exc:
        raise ValueError(f"Unknown model: {model_name}") from exc
    return frame.dropna(subset=list(columns)).reset_index(drop=True)


def make_model(model_name: str, frame: pd.DataFrame, distribution: str = "studentst"):
    """Construct an unfitted model from a clean training frame."""
    r = frame["log_return"]
    if model_name == "B1.1":
        return GARCH11(r, dist=distribution)
    if model_name == "B1.2":
        return GJRGARCH(r, dist=distribution)
    if model_name == "B1.3":
        return EGARCH(r, dist=distribution)
    if model_name == "B2.1":
        return NAGarchNet(r, S=frame["S_t"], dist=distribution)
    columns = {
        "B2.2": ("P_t", "N_t"),
        "B2.3": ("P_mps_surprise", "N_mps_surprise"),
        "B5.3": ("P_t_mps", "N_t_mps"),
        "B5.4": ("P_t_speech", "N_t_speech"),
        "B5.5": ("P_t_bulletins", "N_t_bulletins"),
        "B5.6": ("P_t_t000", "N_t_t000"),
        "B5.7": ("P_t_t080", "N_t_t080"),
    }
    try:
        p_col, n_col = columns[model_name]
    except KeyError as exc:
        raise ValueError(f"Unknown model: {model_name}") from exc
    return NAGarchAsym(r, P=frame[p_col], N=frame[n_col], dist=distribution)
