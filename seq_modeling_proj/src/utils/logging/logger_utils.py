# data_pipeline/logger_utils.py

from utils.logging.logger import setup_logger
from utils.config.config_setup import load_config

config = load_config("baseline_config.json")
logger = setup_logger(log_dir=config['experiment_dirs']['log_dir'], log_file=config['paths']['log_file'])

# def collect_dataloader_info(dataloader, dataset_name="Dataset"):
#     """
#     Collect and print information about a DataLoader.
#     """
#     total_samples = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     batch_size = dataloader.batch_size
    
#     logger.info(f"[bold cyan]{dataset_name} DataLoader Info:[/bold cyan]")
#     logger.info(f"  - Total samples: {total_samples}")
#     logger.info(f"  - Number of batches: {num_batches}")
#     logger.info(f"  - Batch size: {batch_size}")
#     logger.info("-" * 50)
def train_val_data_path():
    """Setup and return a logger with centralized log directory configuration."""
    # p = load_config("baseline_config.json")
    paths = config.get("paths", {})
    train_val_path = paths.get("train_val_data", "")
    return train_val_path

def test_data_path():
    """Setup and return a logger with centralized log directory configuration."""
    # p = load_config("baseline_config.json")
    paths = config.get("paths", {})
    test_data_path = paths.get("test_data", "")
    return test_data_path