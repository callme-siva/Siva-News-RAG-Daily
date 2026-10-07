"""Logging that writes inside the workspace and never contains keys (R3, R11)."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from newsrag.secrets import KEYS, KeyStore, RedactingFilter

LOGGER_NAME = "newsrag"
_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(
    logs_dir: Path | None = None, level: int = logging.INFO, store: KeyStore = KEYS
) -> logging.Logger:
    """Configure the `newsrag` logger. Every handler gets the redacting filter."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if logs_dir is not None:
        logs_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(
            RotatingFileHandler(
                logs_dir / "newsrag.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
            )
        )
    for handler in handlers:
        handler.setFormatter(logging.Formatter(_FORMAT))
        handler.addFilter(RedactingFilter(store))
        logger.addHandler(handler)
    return logger
