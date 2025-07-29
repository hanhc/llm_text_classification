# src/utils/logger_setup.py

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
import os

def setup_logger(log_dir="logs", log_level=logging.INFO):
    """
    Set up the global logger for the project.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file_path = os.path.join(log_dir, "app.log")

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Avoid adding handlers multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create file handler (rotates daily)
    file_handler = TimedRotatingFileHandler(
        log_file_path, when="D", interval=1, backupCount=7, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger