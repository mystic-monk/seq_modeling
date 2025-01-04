"""
Configuration file for setting hyperparameters and logging setup.
"""

from lightning import seed_everything
from lightning.pytorch.loggers import CSVLogger
from torch import nn

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure reproducibility
seed_everything(1, workers=True)

# Hyperparameters

p = dict(
    seq_len=14,  # Sequence length for input data
    batch_size=16,  # Batch size for training
    criterion=nn.MSELoss(),  # Loss function
    max_epochs=500,  # Maximum number of training epochs
    n_features=1,  # Number of input features
    hidden_size=1,  # Number of hidden units in LSTM
    num_layers=256,  # Number of LSTM layers
    num_workers=0,  # Number of data loader workers
    dropout=0.2,  # Dropout rate for regularization
    learning_rate=0.001,  # Learning rate for the optimizer
    output_size=14,  # Update the output size to 14
    run_test=True,  # Run test after training
    log_dir="tb_logs",  # Directory for logs
    experiment_name="experiment",  # Experiment name
    experiment_version="02",  # Experiment version
    metrics_dir="metrics",  # Directory for metrics
    data_path="../../data/transformed/influenza_features.parquet",  # Path to the data file
)

# Logging
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
        "data_path": (str, lambda x: len(x) > 0),
    }
    for key, (expected_type, condition) in required_keys.items():
        if key not in params:
            raise ValueError(f"Missing required parameter: {key}")
        if not isinstance(params[key], expected_type):
            raise ValueError(f"Invalid type for parameter {key}: {type(params[key])}")
        if not condition(params[key]):
            raise ValueError(f"Invalid value for parameter {key}: {params[key]}")

def print_config(params):
    """
    Print the current configuration parameters.
    
    Args:
        params (dict): Dictionary of configuration parameters.
    """
    print("Current Configuration:")
    for key, value in params.items():
        print(f"{key}: {value}")