import os
from utils.config_loader import load_config
from utils.logger import setup_logger

def get_logger():
    """Setup and return a logger with centralized log directory configuration."""
    # Load the configuration
    p = load_config()
    paths = p.get("paths", {})
    
    # Get log directory and log file from config
    log_dir = paths.get("log_dir", "logs")  # Default to "logs" if not in config
    log_file = paths.get("log_file", "logs.log")  # Default to "logs.log" if not in config
    
    # Ensure that the log directory exists
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up the logger and return it
    return setup_logger(log_dir=log_dir, log_file=log_file)


def train_val_data_path():
    """Setup and return a logger with centralized log directory configuration."""
    # Load the configuration
    p = load_config()
    paths = p.get("paths", {})
    
    # Get log directory and log file from config
    train_val_path = paths.get("train_val_data", "")  
    
    
    # Set up the logger and return it
    return train_val_path