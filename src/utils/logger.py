"""
Logger Configuration Module for Fake News Detection Pipeline
=============================================================
This module sets up a basic logging configuration for the entire pipeline.
It defines a consistent log format (timestamp, log level, message) and
provides a simple function to obtain a logger instance.
"""

import logging

# Configure the root logger with:
# - Minimum log level: INFO (shows INFO, WARNING, ERROR, CRITICAL messages)
# - Log format: timestamp, log level name, and the actual log message
# Example output: "2025-04-14 10:30:45,123 - INFO - Loaded 120000 samples"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_logger():
    """
    Retrieve a logger instance for use across pipeline components.

    Returns
    -------
    logging.Logger
        A logger object (root logger) configured with the above settings.
        Use this to log messages at different levels, e.g.:
            logger = get_logger()
            logger.info("Starting data ingestion")
            logger.error("File not found")
    """
    return logging.getLogger()
