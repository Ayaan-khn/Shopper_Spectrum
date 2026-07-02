"""Logging helpers."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from config.settings import LOGS_DIR


def get_logger(name: str) -> logging.Logger:
    """Return a configured project logger."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handler = RotatingFileHandler(LOGS_DIR / "shopper_spectrum.log", maxBytes=1_000_000, backupCount=3)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
