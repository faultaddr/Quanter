"""Structured logging module for QuantTool."""

import logging
import sys
from typing import Dict, Any
from pathlib import Path
from ..config.settings import settings


def setup_logging(
    run_id: str = None, task_id: str = None, symbol: str = None, timeframe: str = None
):
    """Setup structured logging for QuantTool."""

    # Get logging config from settings
    log_level = settings.get("logging.level", "INFO")
    log_format = settings.get(
        "logging.format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    log_file = settings.get("logging.file", "quanttool.log")

    # Create formatter with additional context if provided
    class ContextFilter(logging.Filter):
        def __init__(self, run_id=None, task_id=None, symbol=None, timeframe=None):
            super().__init__()
            self.run_id = run_id
            self.task_id = task_id
            self.symbol = symbol
            self.timeframe = timeframe

        def filter(self, record):
            record.run_id = self.run_id or getattr(record, "run_id", "")
            record.task_id = self.task_id or getattr(record, "task_id", "")
            record.symbol = self.symbol or getattr(record, "symbol", "")
            record.timeframe = self.timeframe or getattr(record, "timeframe", "")
            return True

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(console_handler)

    # File handler
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(file_handler)

    # Add context filter to all loggers
    for name in logging.Logger.manager.loggerDict:
        logging.getLogger(name).addFilter(
            ContextFilter(run_id, task_id, symbol, timeframe)
        )


def get_logger(
    name: str,
    run_id: str = None,
    task_id: str = None,
    symbol: str = None,
    timeframe: str = None,
):
    """Get a logger with optional contextual information."""
    logger = logging.getLogger(name)

    # Apply context filter
    class ContextFilter(logging.Filter):
        def __init__(self, run_id=None, task_id=None, symbol=None, timeframe=None):
            super().__init__()
            self.run_id = run_id
            self.task_id = task_id
            self.symbol = symbol
            self.timeframe = timeframe

        def filter(self, record):
            record.run_id = self.run_id or getattr(record, "run_id", "")
            record.task_id = self.task_id or getattr(record, "task_id", "")
            record.symbol = self.symbol or getattr(record, "symbol", "")
            record.timeframe = self.timeframe or getattr(record, "timeframe", "")
            return True

    # Add the context filter to this specific logger
    logger.addFilter(ContextFilter(run_id, task_id, symbol, timeframe))

    return logger


# Initialize logging with default settings
setup_logging()
