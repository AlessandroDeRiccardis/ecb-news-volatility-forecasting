#!/usr/bin/env python3
"""Run rolling and increasing-window OOS forecasts."""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from ecb_vol_forecasting.config import load_config
from ecb_vol_forecasting.logging_utils import configure_logging
from ecb_vol_forecasting.models.factory import MAIN_MODELS
from ecb_vol_forecasting.pipeline import forecast_models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MAIN_MODELS))
    parser.add_argument("--max-origins", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    log = configure_logging()
    max_origins = 20 if args.quick else args.max_origins
    outputs = forecast_models(load_config(args.config), tuple(args.models), max_origins)
    log.info("Wrote %d forecast files.", len(outputs))


if __name__ == "__main__":
    main()
