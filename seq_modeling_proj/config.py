# config.py
from lightning import seed_everything
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from torch import nn

# Ensure reproducibility
seed_everything(1, workers=True)

# Hyperparameters
p = dict(
    seq_len=14,  # Sequence length for input data
    batch_size=64,  # Batch size for training
    criterion=nn.MSELoss(),  # Loss function
    max_epochs=50,  # Maximum number of training epochs
    n_features=1,  # Number of input features
    hidden_size=7,  # Number of hidden units in LSTM
    num_layers=8,  # Number of LSTM layers
    num_workers=0,  # Number of data loader workers
    dropout=0.2,  # Dropout rate for regularization
    learning_rate=0.001,  # Learning rate for the optimizer
    output_size=1,  # Size of the output layer
    run_test=True, # Example value, set to appropriate value as needed
)

# Logging
tensorboard_logger = TensorBoardLogger(save_dir = "tb_logs", name="experiment", version="02")
csv_logger = CSVLogger(save_dir="tb_logs", name="experiment")

# Parameter validation (optional)
def validate_params(params):
    required_keys = ["seq_len", "batch_size", "criterion", "max_epochs", "n_features",
                     "hidden_size", "num_layers", "num_workers", "dropout", "learning_rate", "output_size"]
    for key in required_keys:
        if key not in params:
            raise ValueError(f"Missing required parameter: {key}")
        if not isinstance(params[key], (int, float, nn.Module)) and key != "criterion":
            raise ValueError(f"Invalid type for parameter {key}: {type(params[key])}")