# ~/seq_modeling_proj/utils/logger.py
import logging
from rich.console import Console
from rich.logging import RichHandler
# from pathlib import Path
import os

# Initialize the rich console
console = Console()

# def setup_logger(phase: str="Training", log_dir: str, log_file: str, log_level=logging.INFO, name="train_logger"):
def setup_logger(log_dir: str, log_file: str, log_level: int = logging.INFO, name: str = "train_logger", phase: str = "Training"):
    """
    Sets up the logger to log to both console and a file.

    Args:
        log_dir (str): Directory to save log files.
        log_file (str): Log file name.
        log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
    
    Returns:
        logger (logging.Logger): Configured logger instance.
    """
    # print(log_dir)
    # print(log_file)
    log_path = log_dir

    logger = logging.getLogger(name)  # Use a consistent name
    logger.setLevel(log_level)

    if logger.hasHandlers():
        return logger 

    # Define log format
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    date_format = "%m/%d/%y %H:%M:%S"

    # Console logging with RichHandler
    rich_handler = RichHandler(console=console, rich_tracebacks=True, markup=True)
    rich_handler.setFormatter(logging.Formatter(fmt=log_format, datefmt=date_format))
    logger.addHandler(rich_handler)

    # File logging with rotation (max 5MB per file, keeps 3 backups)
    log_file_path = os.path.join(log_path, log_file)
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(fmt=log_format, datefmt=date_format))
    logger.addHandler(file_handler)

    # Prevent double logging
    logger.propagate = False

    logger.info(f"{phase} Logging to {log_file_path}")
    
    return logger
