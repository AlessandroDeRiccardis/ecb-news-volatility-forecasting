#!/usr/bin/env python3
"""Estimate the main in-sample model suite."""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from ecb_vol_forecasting.config import load_config
from ecb_vol_forecasting.logging_utils import configure_logging
from ecb_vol_forecasting.models.factory import MAIN_MODELS
from ecb_vol_forecasting.pipeline import estimate_models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MAIN_MODELS))
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    log = configure_logging()
    result = estimate_models(load_config(args.config), tuple(args.models))
    log.info("Wrote in-sample comparison for %d models.", len(result))
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
