# config.py
from lightning import seed_everything
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from torch import nn

# Ensure reproducibility
seed_everything(1, workers=True)

# Hyperparameters
p = dict(
    seq_len=30,  # Sequence length for input data
    batch_size=16,  # Batch size for training
    criterion=nn.MSELoss(),  # Loss function
    max_epochs=50,  # Maximum number of training epochs
    n_features=1,  # Number of input features
    hidden_size=14,  # Number of hidden units in LSTM
    num_layers=8,  # Number of LSTM layers
    num_workers=0,  # Number of data loader workers
    dropout=0.2,  # Dropout rate for regularization
    learning_rate=0.001,  # Learning rate for the optimizer
    output_size=14,  # Size of the output layer
    run_test=True,  # Example value, set to appropriate value as needed
    #model_type="LSTM",  # Model type: LSTM or Hybrid
    log_dir="tb_logs",  # Directory for logs
    experiment_name="experiment",  # Experiment name
    experiment_version="02",  # Experiment version
    metrics_dir="metrics",  # Directory for metrics
)

# Logging
tensorboard_logger = TensorBoardLogger(
    save_dir=p["log_dir"],
    name=p["experiment_name"],
    version=p["experiment_version"],
)
csv_logger = CSVLogger(   
    save_dir=p["log_dir"],
    name=p["experiment_name"],)

# Parameter validation (optional)
def validate_params(params):
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
    }
    for key, (expected_type, condition) in required_keys.items():
        if key not in params:
            raise ValueError(f"Missing required parameter: {key}")
        if not isinstance(params[key], expected_type):
            raise ValueError(f"Invalid type for parameter {key}: {type(params[key])}")
        if not condition(params[key]):
            raise ValueError(f"Invalid value for parameter {key}: {params[key]}")
        
def print_config(params):
    print("Current Configuration:")
    for key, value in params.items():
        print(f"{key}: {value}")