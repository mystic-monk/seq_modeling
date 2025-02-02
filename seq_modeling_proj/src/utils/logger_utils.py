# data_pipeline/logger_utils.py

from utils.logging_setup import get_logger

# Get the logger from the centralized setup
logger = get_logger()

def collect_dataloader_info(dataloader, dataset_name="Dataset"):
    """
    Collect and print information about a DataLoader.
    """
    total_samples = len(dataloader.dataset)
    num_batches = len(dataloader)
    batch_size = dataloader.batch_size
    
    logger.info(f"[bold cyan]{dataset_name} DataLoader Info:[/bold cyan]")
    logger.info(f"  - Total samples: {total_samples}")
    logger.info(f"  - Number of batches: {num_batches}")
    logger.info(f"  - Batch size: {batch_size}")
    logger.info("-" * 50)
