"""
Logging configuration for Chronos-ESM.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "chronos_esm",
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure and return a logger.

    Args:
        name: Logger name.
        log_file: Path to log file.
        level: Logging level.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
