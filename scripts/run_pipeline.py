#!/usr/bin/env python3
"""Run the reproducible modeling pipeline from the processed snapshot."""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from ecb_vol_forecasting.config import load_config
from ecb_vol_forecasting.data import load_master_dataset
from ecb_vol_forecasting.logging_utils import configure_logging
from ecb_vol_forecasting.pipeline import estimate_models, evaluate_saved_forecasts, forecast_models
from ecb_vol_forecasting.plotting import make_core_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use 20 OOS origins.")
    parser.add_argument("--skip-forecasts", action="store_true")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    log = configure_logging()
    config = load_config(args.config)
    frame = load_master_dataset(config.paths.master_dataset)
    make_core_artifacts(frame, config.paths.tables, config.paths.figures)
    estimate_models(config)
    if not args.skip_forecasts:
        forecast_models(config, max_origins=20 if args.quick else None)
        evaluate_saved_forecasts(config)
    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
