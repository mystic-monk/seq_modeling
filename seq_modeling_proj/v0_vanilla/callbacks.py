# callbacks.py
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping, Callback, ModelCheckpoint
import os
import csv
import time  # Import the time module
import torch  # Import torch module

class CSVLoggerCallback(Callback):
    """
    Custom callback to log metrics into a CSV file for manual extraction.
    """
    def __init__(self, log_file: str = "metrics.csv", metrics_to_log: list[str] = None):
        super().__init__()
        self.log_file = log_file
        self.metrics_to_log = metrics_to_log or [
            "experiment_id", "learning_rate", "batch_size", "epochs",
            "train_loss", "val_loss", "test_loss", "test_mae", "test_mse",
            "test_rmse", "test_r2", "test_mape"
        ]
        self._initialize_csv()

    def _initialize_csv(self):
        """
        Initialize the CSV file with headers.
        """
        with open(self.log_file, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(self.metrics_to_log)  # Write the header row
    
    def _write_row(self, log_data):
        """
        Append a row of data to the CSV file.
        """
        try:
            with open(self.log_file, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                # Convert tensor values to floats and handle missing metrics
                log_data = [value.item() if isinstance(value, torch.Tensor) else value for value in log_data]
                log_data = [value if value is not None else '' for value in log_data]
                writer.writerow(log_data)
        except IOError as e:
            print(f"Error writing to log file {self.log_file}: {e}")

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        Logs metrics at the end of each validation step.
        """
        metrics = trainer.callback_metrics
        log_data = [
            trainer.current_epoch,
            pl_module.hparams.learning_rate,
            pl_module.hparams.batch_size,
            trainer.current_epoch,
            metrics.get("train_loss", None),
            metrics.get("val_loss", None),
            metrics.get("test_loss", None),
            metrics.get("test_mae", None),
            metrics.get("test_mse", None),
            metrics.get("test_rmse", None),
            metrics.get("test_r2", None),
            metrics.get("test_mape", None)
        ]
        log_data = [value.item() if isinstance(value, torch.Tensor) else value for value in log_data]
        log_data = [value if value is not None else '' for value in log_data]
        self._write_row(log_data)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """
        Logs metrics at the end of each training batch.
        """
        metrics = trainer.callback_metrics
        log_data = [
            trainer.current_epoch,
            pl_module.hparams.learning_rate,
            pl_module.hparams.batch_size,
            trainer.current_epoch,
            metrics.get("train_loss", None),
            metrics.get("val_loss", None),
            metrics.get("test_loss", None),
            metrics.get("test_mae", None),
            metrics.get("test_mse", None),
            metrics.get("test_rmse", None),
            metrics.get("test_r2", None),
            metrics.get("test_mape", None)
        ]
        log_data = [value.item() if isinstance(value, torch.Tensor) else value for value in log_data]
        log_data = [value if value is not None else '' for value in log_data]
        self._write_row(log_data)

class PrintingCallback(Callback):
    """
    Custom callback to print training progress and key metrics to the console.
    """
    def __init__(self, verbose=True):
        """
        Args:
            verbose (bool): Whether to print detailed logs.
        """
        super().__init__()
        self.verbose = verbose
        self.min_val_loss = float("inf")

    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()
        if self.verbose:
            print("Training has started!")

    def on_train_epoch_end(self, trainer, pl_module):
        print(f"Epoch {trainer.current_epoch}: train_loss = {trainer.callback_metrics['train_loss']}, val_loss = {trainer.callback_metrics['val_loss']}, Min val_loss = {self.min_val_loss}")

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx: int
    ) -> None:
        if self.verbose:
            print(f"Epoch {trainer.current_epoch} | Batch {batch_idx} completed. | Train Loss: {trainer.callback_metrics['train_loss']:.4f}", end="\r")  

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        total_time = time.time() - self.start_time
        print(f"Training completed in {total_time:.2f} seconds.")

    def on_test_start(self, trainer, pl_module):
        print("Testing has started!")

    def on_test_end(self, trainer, pl_module):
        print("Testing has completed!")
    
    def test_step_end(self, trainer, pl_module):
        print("Test step completed!")
    
    def on_validation_epoch_end(self, trainer, pl_module):
        
        if trainer.callback_metrics["val_loss"] < self.min_val_loss:
            self.min_val_loss = trainer.callback_metrics["val_loss"]
            

early_stop_callback = EarlyStopping(
    monitor="val_loss",  # Monitor validation loss
    min_delta=0.001,  # Small delta to avoid early stopping on minor fluctuations
    patience=10,  # Stop after 10 epochs of no improvement
    verbose=False,  # Do not print early stopping messages
    mode="min",  # Minimize validation loss
)

# Ensure the checkpoint directory exists
checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Set up ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",  # Criterion to monitor (e.g., validation loss)
    dirpath=checkpoint_dir,  # Save checkpoints in the specified directory
    filename="model-{epoch:02d}-{val_loss:.2f}",  # Filename format
    save_top_k=1,  # Save the best model
    mode="min",  # Save the minimum validation loss
    save_last=False,  # Save the latest model
    auto_insert_metric_name=False,  # Do not insert the metric name in the filename
    verbose=False,  # Suppress the warning message
)