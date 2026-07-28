"""Consistent logging configuration for all CSA-YOLO scripts and modules."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logger(level: str = "INFO", log_file: Path | None = None) -> logging.Logger:
    """Configure the root ``csa_yolo`` logger.

    Args:
        level: Logging level name (``DEBUG``, ``INFO``, ...).
        log_file: Optional file to also write logs to (created if missing).

    Returns:
        The configured ``csa_yolo`` package logger.
    """
    logger = logging.getLogger("csa_yolo")
    logger.setLevel(level.upper())
    logger.handlers.clear()

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(logging.Formatter(_FORMAT))
    logger.addHandler(stream)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(file_handler)

    return logger
