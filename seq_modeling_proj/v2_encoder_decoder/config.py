"""
Configuration file for setting hyperparameters and logging setup.
"""
import os
import logging
from lightning import seed_everything
from lightning.pytorch.loggers import CSVLogger
from torch import nn
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
import torch
import numpy as np
import random

# Initialize the rich console
console = Console()

# Ensure reproducibility
seed_everything(1, workers=True)

seed = 303
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
# Ensure deterministic behavior (for reproducibility)
torch.backends.cudnn.deterministic = True

# Disable cuDNN benchmark for deterministic results (especially with variable input sizes)
torch.backends.cudnn.benchmark = False

# Hyperparameters
p = dict(
    seq_len=14,  
    batch_size=14,  
    criterion=nn.MSELoss(),  
    max_epochs=200,  
    n_features=6,  
    data_columns=[
        "event_creation_date", "log_cases_14d_moving_avg",
        "lag_1d", "lag_7d", "lag_14d",
        "weekday_sin", "annual_sin"],
    hidden_size=8,  
    num_layers=7,  
    num_workers=0,  
    dropout=0.2,  
    learning_rate=0.3, 
    output_size=1,  
    run_test=True, 
    log_dir="logs",
    log_file="logs.log",
    experiment_name="experiment",
    experiment_version="02",
    metrics_dir="metrics",
    data_path="../../data/transformed/influenza_features.parquet", 
    train_val_data_path="../../data/transformed/influenza_features_train_val.parquet",  # Path to the data file
    test_data_path="../../data/transformed/influenza_features_test.parquet",
    nums_splits=1,
    rnn_type="LSTM", 
    initial=14,
    horizon=14,
    period=14, 
    debug=True, 
)

# Logger setup
def setup_logger(log_dir, log_file, log_level=logging.INFO):
    """
    Sets up the logger to log to both console and a file.

    Args:
        log_dir (str): Directory to save log files.
        log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    # Define the log format (same for both console and file)
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    date_format = "%m/%d/%y %H:%M:%S"

    # Console handler with RichHandler for colored output
    rich_handler = RichHandler(console=console, rich_tracebacks=True)
    rich_handler.setFormatter(logging.Formatter(fmt=log_format, datefmt=date_format, ))
    logger.addHandler(rich_handler)

    # File handler for logging to file with UTF-8 encoding
    log_fil_path = os.path.join(log_dir, log_file)
    file_handler = logging.FileHandler(log_fil_path, encoding='utf-8')
    # file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    file_handler.setFormatter(logging.Formatter(fmt=log_format, datefmt=date_format))
    logger.addHandler(file_handler)

    # Prevent double logging
    logger.propagate = False
    return logger

# Ensure metrics directory exists
os.makedirs(p["metrics_dir"], exist_ok=True)

# Ensure logging directory exists
os.makedirs(p["log_dir"], exist_ok=True)

logger = setup_logger(p["log_dir"], p["log_file"], log_level=logging.DEBUG)

# CSV Logger
csv_logger = CSVLogger(   
    save_dir=p["log_dir"],
    name=p["experiment_name"],)

# Parameter validation (optional)
def validate_params(params):
    """
    Validate the configuration parameters.
    
    Args:
        params (dict): Dictionary of configuration parameters.
    
    Raises:
        ValueError: If any required parameter is missing or invalid.
    """
    required_keys = {
        "seq_len": (int, lambda x: x > 0),
        "batch_size": (int, lambda x: x > 0),
        "criterion": (nn.Module, lambda x: True),
        "max_epochs": (int, lambda x: x > 0),
        "n_features": (int, lambda x: x > 0),
        "hidden_size": (int, lambda x: x > 0),
        "num_layers": (int, lambda x: x > 0),
        "num_workers": (int, lambda x: x >= 0),
        "dropout": (float, lambda x: 0 <= x <= 1),
        "learning_rate": (float, lambda x: x > 0),
        "output_size": (int, lambda x: x > 0),
        "train_val_data_path": (str, lambda x: len(x) > 0),
        "test_data_path": (str, lambda x: len(x) > 0),
        "nums_splits": (int, lambda x: x >= 1),
    }
    errors = []
    for key, (expected_type, condition) in required_keys.items():
        if key not in params:
            errors.append(f"Missing required parameter: {key}")
        elif not isinstance(params[key], expected_type):
            errors.append(f"Invalid type for parameter {key}: Expected {expected_type.__name__}, got {type(params[key]).__name__}")
        elif not condition(params[key]):
            errors.append(f"Invalid value for parameter {key}: {params[key]}")

    if errors:
        table = Table(title="Parameter Validation Errors")
        table.add_column("Error", justify="left", style="bold red")
        for error in errors:
            table.add_row(error)
        console.print(table)
        raise ValueError("Parameter validation failed. See errors above.")

validate_params(p)


logger.info("Configuration and logging setup complete.")