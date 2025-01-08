"""
Configuration file for setting hyperparameters and logging setup.
"""
import os
import logging
from lightning import seed_everything
from lightning.pytorch.loggers import CSVLogger
from torch import nn
from rich.console import Console
from rich.table import Table

# Logging setup
logging.basicConfig(
    level=logging.INFO,  # Ensure this is set to INFO or DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.propagate = False

# Add FileHandler for logging to a file
file_handler = logging.FileHandler("app.log")
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Ensure reproducibility
seed_everything(1, workers=True)

# Hyperparameters
p = dict(
    seq_len=14,  # Sequence length for input data
    batch_size=16,  # Batch size for training
    criterion=nn.MSELoss(),  # Loss function
    max_epochs=100,  # Maximum number of training epochs
    n_features=1,  # Number of input features
    hidden_size=16,  # Number of hidden units in LSTM
    num_layers=2,  # Number of LSTM layers
    num_workers=0,  # Number of data loader workers
    dropout=0.2,  # Dropout rate for regularization
    learning_rate=0.1,  # Learning rate for the optimizer
    output_size=14,  # Update the output size to 14
    run_test=True,  # Run test after training
    log_dir="logs",  # Directory for logs
    experiment_name="experiment",  # Experiment name
    experiment_version="02",  # Experiment version
    metrics_dir="metrics",  # Directory for metrics
    train_val_data_path="../../data/transformed/influenza_features_train_val.parquet",  # Path to the data file
    test_data_path="../../data/transformed/influenza_features_test.parquet",
    nums_splits=2,  # Number of splits for walk forward validation 
    debug=False,  # Debug mode
)

# Ensure metrics directory exists
os.makedirs(p["metrics_dir"], exist_ok=True)

# Ensure logging directory exists
os.makedirs(p["log_dir"], exist_ok=True)
# Logger setup
def setup_logger(log_dir, log_level=logging.INFO):
    """
    Sets up the logger to log to both console and a file.

    Args:
        log_dir (str): Directory to save log files.
        log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)

    # File handler
    log_file = os.path.join(log_dir, "logs.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger

logger = setup_logger(p["log_dir"], log_level=logging.DEBUG)
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
    for key, (expected_type, condition) in required_keys.items():
        if key not in params:
            logger.error(f"Missing required parameter: {key}")
            raise ValueError(f"Missing required parameter: {key}")
        if not isinstance(params[key], expected_type):
            logger.error(f"Invalid type for parameter {key}: {type(params[key])}")
            raise ValueError(f"Invalid type for parameter {key}: {type(params[key])}")
        if not condition(params[key]):
            logger.error(f"Invalid value for parameter {key}: {params[key]}")
            raise ValueError(f"Invalid value for parameter {key}: {params[key]}")

validate_params(p)

logger.info("Configuration and logging setup complete.")