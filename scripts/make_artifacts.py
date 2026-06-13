#!/usr/bin/env python3
"""Build figures and evaluate available forecast files."""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from ecb_vol_forecasting.config import load_config
from ecb_vol_forecasting.data import load_master_dataset
from ecb_vol_forecasting.logging_utils import configure_logging
from ecb_vol_forecasting.pipeline import evaluate_saved_forecasts
from ecb_vol_forecasting.plotting import make_core_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    log = configure_logging()
    config = load_config(args.config)
    frame = load_master_dataset(config.paths.master_dataset)
    make_core_artifacts(frame, config.paths.tables, config.paths.figures)
    log.info("Wrote descriptive table and core figures.")
    if not args.skip_evaluation:
        table = evaluate_saved_forecasts(config)
        log.info("Evaluated %d model/scheme/horizon cells.", len(table))


if __name__ == "__main__":
    main()
