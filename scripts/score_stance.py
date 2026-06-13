#!/usr/bin/env python3
"""Run the archived FOMC-RoBERTa sentence scorer."""

from __future__ import annotations

import argparse

import _bootstrap

from ecb_vol_forecasting.data.legacy import run_legacy_script
from ecb_vol_forecasting.logging_utils import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    forwarded = []
    if args.limit:
        forwarded.extend(["--limit", str(args.limit)])
    if args.resume:
        forwarded.append("--resume")
    configure_logging().warning(
        "Stance scoring requires output/ecb_documents_master.csv and cleaned raw texts. "
        "These inputs are not included in the repository snapshot."
    )
    run_legacy_script(_bootstrap.ROOT, "03a_score_sentences.py", forwarded)


if __name__ == "__main__":
    main()
