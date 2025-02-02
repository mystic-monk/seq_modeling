# ~/seq_modeling_proj/utils/logger.py
import logging
from rich.console import Console
from rich.logging import RichHandler
import os

# Initialize the rich console
console = Console()

def setup_logger(log_dir: str, log_file: str, log_level=logging.INFO):
    """
    Sets up the logger to log to both console and a file.

    Args:
        log_dir (str): Directory to save log files.
        log_file (str): Log file name.
        log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
    
    Returns:
        logger (logging.Logger): Configured logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)  # Ensure log directory exists

    logger = logging.getLogger("app_logger")  # Use a consistent name
    logger.setLevel(log_level)

    if logger.hasHandlers():
        return logger 

    # Define log format
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    date_format = "%m/%d/%y %H:%M:%S"

    # 1️⃣ Console logging with RichHandler
    rich_handler = RichHandler(console=console, rich_tracebacks=True, markup=True)
    rich_handler.setFormatter(logging.Formatter(fmt=log_format, datefmt=date_format))
    logger.addHandler(rich_handler)

   # 2️⃣ File logging with rotation (max 5MB per file, keeps 3 backups)
    log_file_path = os.path.join(log_dir, log_file)
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(fmt=log_format, datefmt=date_format))
    logger.addHandler(file_handler)

    # Prevent double logging
    logger.propagate = False
    
    return logger
