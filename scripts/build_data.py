#!/usr/bin/env python3
"""Validate the processed snapshot or run archived raw-data stages."""

from __future__ import annotations

import argparse

import _bootstrap

from ecb_vol_forecasting.config import load_config
from ecb_vol_forecasting.data import load_master_dataset
from ecb_vol_forecasting.data.legacy import run_legacy_script
from ecb_vol_forecasting.logging_utils import configure_logging

ACQUISITION_STAGES = (
    "01a_download_market_data.py",
    "01b_download_controls.py",
    "02_ecb_collection.py",
)
AGGREGATION_STAGES = (
    "04a_build_daily_series.py",
    "05a_prepare_master_dataset.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--legacy-acquisition",
        action="store_true",
        help="Run archived market and ECB document acquisition stages.",
    )
    parser.add_argument(
        "--legacy-aggregate",
        action="store_true",
        help="Aggregate existing legacy sentence scores and build the master.",
    )
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    log = configure_logging()
    config = load_config(args.config)
    if not (args.legacy_acquisition or args.legacy_aggregate):
        frame = load_master_dataset(config.paths.master_dataset)
        log.info("Processed snapshot is valid: %d rows x %d columns.", *frame.shape)
        return
    stages = ACQUISITION_STAGES if args.legacy_acquisition else AGGREGATION_STAGES
    for filename in stages:
        log.info("Running archived stage: %s", filename)
        run_legacy_script(_bootstrap.ROOT, filename)
    log.warning(
        "Archived stages write legacy-compatible market_data/ and output/ directories. "
        "Stance scoring is a separate expensive stage; review outputs before promotion."
    )


if __name__ == "__main__":
    main()
