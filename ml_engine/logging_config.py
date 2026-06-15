"""
AutoML Studio — Structured Logging Configuration

Provides centralized logging setup with:
- Console handler with colored level names
- Rotating file handler (5MB, 3 backups) to logs/automl.log
- Configurable log level via environment variable AUTOML_LOG_LEVEL
"""

import os
import logging
from logging.handlers import RotatingFileHandler


LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LOG_LEVEL = "INFO"


def setup_logging(log_dir=None, log_level=None):
    """
    Configure root logger with console + rotating file handlers.
    
    Args:
        log_dir: Directory for log files. Defaults to 'logs/' in project root.
        log_level: Logging level string (DEBUG, INFO, WARNING, ERROR).
                   Defaults to AUTOML_LOG_LEVEL env var or INFO.
    """
    level_str = log_level or os.environ.get("AUTOML_LOG_LEVEL", DEFAULT_LOG_LEVEL)
    level = getattr(logging, level_str.upper(), logging.INFO)
    
    root_logger = logging.getLogger()
    
    # Avoid duplicate handlers on repeated calls
    if root_logger.handlers:
        return
    
    root_logger.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    
    # Console handler
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    root_logger.addHandler(console)
    
    # Rotating file handler
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    
    try:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "automl.log")
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        root_logger.warning(f"Could not create file log handler: {e}")
    
    # Silence noisy third-party loggers
    for noisy in ["urllib3", "google", "httpcore", "httpx", "optuna"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    
    root_logger.info("AutoML Studio logging initialized (level=%s)", level_str)
