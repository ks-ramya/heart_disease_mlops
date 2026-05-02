"""Structured JSON logging for the API.

Uses ``python-json-logger`` if available, otherwise falls back to a simple
human-readable formatter. The configured logger is named ``heart_disease_api``.
"""

from __future__ import annotations

import logging
import sys


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("heart_disease_api")
    if logger.handlers:  # idempotent
        return logger
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    try:
        from pythonjsonlogger import jsonlogger  # type: ignore
        fmt = jsonlogger.JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
            rename_fields={"asctime": "timestamp", "levelname": "level"},
        )
    except ImportError:  # pragma: no cover - fallback
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
