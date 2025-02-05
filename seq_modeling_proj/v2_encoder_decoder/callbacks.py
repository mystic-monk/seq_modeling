# callbacks.py
# Standard Library Imports
import csv
import os
import time

# Third-Party Library Imports
import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from torchmetrics.regression import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError, R2Score
from rich.console import Console
from rich.text import Text

# Local Module Imports
from config import logger



console = Console()

class CSVLoggerCallback(Callback):
    """
    Custom callback to log metrics into a CSV file for manual extraction.
    """
    def __init__(self, 
                 train_log_file="train_metrics.csv", 
                 val_log_file="val_metrics.csv", 
                 test_log_file="test_metrics.csv",
                train_predictions_file="train_predictions.csv",
                 val_predictions_file="val_predictions.csv",
                 test_predictions_file="test_predictions.csv"):
        super().__init__()
        self.train_log_file = train_log_file
        self.val_log_file = val_log_file
        self.test_log_file = test_log_file
        self.train_predictions_file = train_predictions_file
        self.val_predictions_file = val_predictions_file
        self.test_predictions_file = test_predictions_file

        # Initialize each file with appropriate headers
        self._initialize_csv(self.train_log_file, ["epoch", "learning_rate", "batch_size", 
                                                   "train_loss", "train_r2", "train_mse", "train_rmse", "train_mae", "train_residuals_mean"])
        self._initialize_csv(self.val_log_file, ["epoch", "learning_rate", "batch_size", 
                                                 "val_loss", "val_r2", "val_mse", "val_rmse", "val_mae", "val_residuals_mean"])
        # experimane to be included in test csv
        self._initialize_csv(self.test_log_file, ["test_loss", "test_r2", "test_r2_adj", "test_mse", "test_rmse", "test_mae", "test_residuals_mean"])

        # Initialize predictions CSVs
        self._initialize_csv(self.train_predictions_file, ["epoch", "index", "y", "y_hat"])
        self._initialize_csv(self.val_predictions_file, ["epoch", "index", "y", "y_hat"])
        self._initialize_csv(self.test_predictions_file, ["epoch", "index", "y", "y_hat"])

        super(CSVLoggerCallback, self).__init__()
        
        # Initialize metrics
        self.mae = MeanAbsoluteError()
        self.mse = MeanSquaredError()
        self.r2 = R2Score()
        self.mape = MeanAbsolutePercentageError()

    def _initialize_csv(self, log_file, headers):
        """
        Initialize the CSV file with headers.
        """
        with open(log_file, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(headers)
        
    
    def _write_row(self, log_file, log_data):
        """
        Append a row of data to the specified CSV file.
        """
        try:
            with open(log_file, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                log_data = [value.item() if isinstance(value, torch.Tensor) else value for value in log_data]
                log_data = [value if value is not None else '' for value in log_data]
                writer.writerow(log_data)
        except IOError as e:
            logger.error(f"Error writing to log file {log_file}: {e}")


    def _log_predictions(self, file, epoch, y_all, y_hat_all, use_index=False):
        for idx, (y, y_hat) in enumerate(zip(y_all.flatten().tolist(), y_hat_all.flatten().tolist())):
            self._write_row(file, [epoch, idx, y, y_hat])


    def on_train_epoch_end(self, trainer, pl_module, outputs=None, batch=None, batch_idx=None):
        """
        Logs metrics at the end of each training epoch.
        """
        metrics = trainer.callback_metrics
        
        train_y_all = torch.cat(pl_module.train_epoch_y, dim=0) 
        train_y_hat_all = torch.cat(pl_module.train_epoch_y_hat, dim=0)

        train_r2 = self.r2(train_y_hat_all, train_y_all)
        train_mse = self.mse(train_y_hat_all, train_y_all)
        train_rmse = torch.sqrt(train_mse)
        train_mae = self.mae(train_y_hat_all, train_y_all)
        train_residuals = train_y_all - train_y_hat_all
        test_residuals_mean = torch.mean(train_residuals)

        log_data = [
            trainer.current_epoch,
            pl_module.hparams.learning_rate,
            pl_module.hparams.batch_size,
            metrics.get("train_loss", None),
            train_r2,
            train_mse,
            train_rmse,
            train_mae,
            test_residuals_mean,
        ]

        self._write_row(self.train_log_file, log_data)

        # Log predictions
        self._log_predictions(self.train_predictions_file, trainer.current_epoch, train_y_all, train_y_hat_all)




    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        Logs metrics at the end of each validation epoch.
        """
        metrics = trainer.callback_metrics
        
        val_y_all = torch.cat(pl_module.val_epoch_y, dim=0)        # Concatenate all y tensors
        val_y_hat_all = torch.cat(pl_module.val_epoch_y_hat, dim=0)
        val_r2 = self.r2(val_y_hat_all, val_y_all)
        val_mse = self.mse(val_y_hat_all, val_y_all)
        val_rmse = torch.sqrt(val_mse)
        val_mae = self.mae(val_y_hat_all, val_y_all)
        val_residuals = val_y_all - val_y_hat_all
        val_residual_mean = torch.mean(val_residuals)

        log_data = [
            trainer.current_epoch,
            pl_module.hparams.learning_rate,
            pl_module.hparams.batch_size,
            metrics.get("val_loss", None),
            val_r2,
            val_mse,
            val_rmse,
            val_mae,
            val_residual_mean,
        ]

        self._write_row(self.val_log_file, log_data)
        # Log predictions
        self._log_predictions(self.val_predictions_file, trainer.current_epoch, val_y_all, val_y_hat_all)



    def on_test_epoch_end(self, trainer, pl_module):
        """
        Logs metrics at the end of the test phase.
        """
        
        metrics = trainer.callback_metrics
        
        test_y_all = torch.cat(pl_module.test_epoch_y, dim=0)        # Concatenate all y tensors
        test_y_hat_all = torch.cat(pl_module.test_epoch_y_hat, dim=0)
        
        # print(f"CALLBACKS:: CSV > on_test_epoch_end >>y shape {test_y_all.shape}")
        test_r2 = self.r2(test_y_hat_all, test_y_all)
        test_mse = self.mse(test_y_hat_all, test_y_all)
        test_rmse = torch.sqrt(test_mse)
        test_mae = self.mae(test_y_hat_all, test_y_all)
        test_residuals = test_y_all - test_y_hat_all
        test_residual_mean = torch.mean(test_residuals)

        log_data = [
        
            metrics.get("test_loss", None),
            test_r2,
            metrics.get("test_r2_adj", None),
            test_mse,
            test_rmse,
            test_mae,
            test_residual_mean,
        ]
        self._write_row(self.test_log_file, log_data)

        # Log predictions
        self._log_predictions(self.test_predictions_file, 0, test_y_all, test_y_hat_all, use_index=True)


class PrintingCallback(Callback):
    """
    Custom callback to print training progress and key metrics to the console using rich.
    """
    def __init__(self, verbose=True):
        """
        Args:
            verbose (bool): Whether to print detailed logs.
        """
        super().__init__()
        self.verbose = verbose
        self.min_val_loss = float("inf")
        self.first_val_call = True
        self.start_time = None

    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()
        logger.info("🚀 Training has started!")  # Remove rich formatting for logger

    def on_train_epoch_end(self, trainer, pl_module):

     
        train_loss = trainer.callback_metrics.get("train_loss", "N/A")
        val_loss = trainer.callback_metrics.get("val_loss", "N/A")

        # Prepare the text with color for each metric
        epoch_num = trainer.current_epoch
        epoch_num_text = Text(f"Epoch: {epoch_num:.0f}" if epoch_num != "N/A" else "Epoch: N/A", style="bold cyan")
        train_loss_text = Text(f"Train Loss: {train_loss:.4f}" if train_loss != "N/A" else "Train Loss: N/A", style="bold magenta")
        val_loss_text = Text(f"Validation Loss: {val_loss:.4f}" if val_loss != "N/A" else "Validation Loss: N/A", style="bold green")
        best_val_loss_text = Text(f"Best Validation Loss: {self.min_val_loss:.4f}", style="bold yellow")

        # Combine all the texts into a single line
        console.print(epoch_num_text, train_loss_text, val_loss_text, best_val_loss_text)

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx: int
    ) -> None:
        if self.verbose:
            batch_loss = trainer.callback_metrics.get("batch_loss", 0)
            console.print(
                f"🔄 [bold magenta]Epoch {trainer.current_epoch}[/bold magenta] | "
                f"[cyan]Batch {batch_idx}[/cyan] | "
                f"[yellow]Loss: {batch_loss:.4f}[/yellow]",
                end="\r"
            )

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        total_time = time.time() - self.start_time
        console.print(f"[bold green]✅ Training completed in {total_time:.2f} seconds.[/bold green]")

    def on_test_start(self, trainer, pl_module):
        console.print("[bold blue]🔍 Testing has started![/bold blue]")

    def on_test_end(self, trainer, pl_module):
        console.print("[bold green]🏁 Testing completed![/bold green]")
    
    def on_test_step_end(self, trainer, pl_module):
        console.print("[bold yellow]✅ Test step completed![/bold yellow]")
    
    def on_validation_epoch_end(self, trainer, pl_module):
        
        val_loss = trainer.callback_metrics.get("val_loss", float("inf"))
        if val_loss < self.min_val_loss and not self.first_val_call:
            self.min_val_loss = val_loss
        elif self.first_val_call:
            self.first_val_call = False

early_stop_callback = EarlyStopping(
    monitor="val_loss",  
    min_delta=0.001, 
    patience=10,  
    verbose=False,  
    mode="min",  
)

# Ensure the checkpoint directory exists
checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Set up ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=checkpoint_dir,
    filename="model-{epoch:02d}-{val_loss:.2f}", 
    save_top_k=1,  
    mode="min",  
    save_last=False,  
    auto_insert_metric_name=False,  
    verbose=False,  
)