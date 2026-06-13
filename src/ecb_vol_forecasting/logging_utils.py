"""Consistent command-line logging."""

from __future__ import annotations

import logging


def configure_logging(verbose: bool = False) -> logging.Logger:
    """Configure and return the package logger."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        force=True,
    )
    return logging.getLogger("ecb_vol_forecasting")
