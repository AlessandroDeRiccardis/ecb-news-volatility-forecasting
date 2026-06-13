"""High-level modeling pipeline used by command-line scripts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import ProjectConfig
from .data.schema import load_master_dataset
from .evaluation.comparison import (
    dm_table,
    forecast_combination_table,
    model_confidence_set_table,
)
from .evaluation.reporting import accuracy_table, load_forecasts
from .models.factory import MAIN_MODELS, clean_training_data, is_news_augmented, make_model
from .models.forecasting import run_forecasts


def estimate_models(
    config: ProjectConfig,
    model_names: tuple[str, ...] = MAIN_MODELS,
) -> pd.DataFrame:
    """Fit the main suite in-sample and save a compact comparison table."""
    config.paths.ensure_output_dirs()
    df = load_master_dataset(config.paths.master_dataset)
    rows = []
    for model_name in model_names:
        sample = clean_training_data(df[df["period"] == "insample"], model_name)
        model = make_model(model_name, sample, config.distribution)
        model.fit(n_restarts=2) if is_news_augmented(model_name) else model.fit()
        rows.append(
            {
                "model": model_name,
                "distribution": config.distribution,
                "n_obs": len(sample),
                "loglik": model.loglik,
                "aic": model.aic,
                "n_params": len(model.params),
            }
        )
    result = pd.DataFrame(rows).sort_values("aic")
    result.to_csv(config.paths.tables / "in_sample_model_comparison.csv", index=False)
    return result


def forecast_models(
    config: ProjectConfig,
    model_names: tuple[str, ...] = MAIN_MODELS,
    max_origins: int | None = None,
) -> list[Path]:
    """Run and save the configured OOS forecast suite."""
    config.paths.ensure_output_dirs()
    df = load_master_dataset(config.paths.master_dataset)
    outputs = []
    for model_name in model_names:
        for scheme in config.schemes:
            for horizon in config.horizons:
                forecasts = run_forecasts(
                    df,
                    model_name=model_name,
                    scheme=scheme,
                    horizon=horizon,
                    distribution=config.distribution,
                    step=config.forecast_step,
                    max_origins=max_origins,
                )
                path = config.paths.forecasts / f"{model_name}_{scheme}_h{horizon}.csv"
                forecasts.to_csv(path, index=False)
                outputs.append(path)
    return outputs


def evaluate_saved_forecasts(config: ProjectConfig) -> pd.DataFrame:
    """Evaluate all saved forecasts and write core comparison tables."""
    forecasts = load_forecasts(config.paths.forecasts)
    table = accuracy_table(forecasts)
    table.to_csv(config.paths.tables / "oos_accuracy.csv", index=False)
    models = tuple(sorted(forecasts["model"].unique()))
    dm_table(forecasts, models).to_csv(config.paths.tables / "oos_dm_tests.csv", index=False)
    forecast_combination_table(forecasts).to_csv(
        config.paths.tables / "forecast_combination.csv", index=False
    )
    if forecasts.groupby(["scheme", "horizon"]).size().min() >= 30:
        model_confidence_set_table(forecasts, models, seed=config.random_seed).to_csv(
            config.paths.tables / "oos_mcs.csv", index=False
        )
    return table
