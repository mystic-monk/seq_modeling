# callbacks.py
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping, Callback, ModelCheckpoint
import os
import csv
class CSVLoggerCallback(Callback):
    """
    Custom callback to log metrics into a CSV file for manual extraction.
    """
    def __init__(self, log_file="metrics.csv", metrics_to_log=None):
        super().__init__()
        self.log_file = log_file
        self.metrics_to_log = metrics_to_log or ["epoch", "train_loss", "val_loss", "val_mae"]
        with open(self.log_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(self.metrics_to_log)  # Define headers

    def on_validation_end(self, trainer, pl_module):
        """
        Logs metrics at the end of each validation step.
        """
        metrics = trainer.callback_metrics
        # epoch = trainer.current_epoch
        log_data = [trainer.current_epoch]  

        # Collect specified metrics
        for metric in self.metrics_to_log[1:]:  # Skip "epoch" as it's already added
            log_data.append(metrics.get(metric, None))

        # Append metrics to the CSV file
        try:
            with open(self.log_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(log_data)
        except IOError as e:
            print(f"Error writing to log file {self.log_file}: {e}")
        # train_loss = metrics.get("train_loss", None)
        # val_loss = metrics.get("val_loss", None)
        # val_mae = metrics.get("val_mae", None)

        # with open(self.log_file, mode="a", newline="") as file:
        #     writer = csv.writer(file)
        #     writer.writerow([epoch, train_loss, val_loss, val_mae])


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

    def on_train_start(self, trainer, pl_module):
        if self.verbose:
            print("Training has started!")

    def on_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.verbose:
            print(f"Epoch {trainer.current_epoch} completed.")
            val_loss = trainer.callback_metrics.get("val_loss")
            val_mae = trainer.callback_metrics.get("val_mae")
            if val_loss is not None:
                print(f"Validation loss: {val_loss:.4f}")
            if val_mae is not None:
                print(f"Validation MAE: {val_mae:.4f}")


early_stop_callback = EarlyStopping(
    monitor="val_loss",  # Monitor validation loss
    min_delta=0.001,  # Small delta to avoid early stopping on minor fluctuations
    patience=10,  # Stop after 3 epochs of no improvement
    verbose=True,
    mode="min",  # Minimize validation loss
)

# Ensure the checkpoint directory exists
checkpoint_dir = "./checkpoints"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Set up ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",  # Criterion to monitor (e.g., validation loss)
    dirpath=checkpoint_dir,  # Directory to save the checkpoints
    filename="model-{epoch:02d}-{val_loss:.2f}",  # File name format
    save_top_k=1,  # Save the top 1 model based on the monitored criterion
    mode="min",  # 'min' or 'max' depending on whether you want to minimize or maximize the criterion
    save_last=True,  # Optionally, save the last model
    auto_insert_metric_name=False,  # Avoid redundancy in filenames
)
