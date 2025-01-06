# train.py
import os
import mlflow
import mlflow.pytorch
from datetime import datetime
from data_set import LineListingDataModule
from lightning import Trainer
from model import LSTMRegressor
from config import p, validate_params, csv_logger, logger
from callbacks import PrintingCallback, early_stop_callback, checkpoint_callback, CSVLoggerCallback
import warnings
import torch
from torchinfo import summary

# Suppress specific warnings for cleaner output
warnings.filterwarnings(
    "ignore",
    message="The '.*dataloader.*' does not have many workers.*",
    category=UserWarning,
)

def configure_callbacks(metrics_dir):
    """
    Configure callbacks for the trainer.
    """
    return [
        PrintingCallback(verbose=True),
        early_stop_callback,
        checkpoint_callback,
        CSVLoggerCallback(log_file=os.path.join(metrics_dir, "training_metrics.csv")),
    ]

def main():
    """
    Main script for training and testing the LSTM model on time series data.
    Logs hyperparameters, metrics, and model artifacts using MLflow.
    """
    # Validate configuration parameters
    validate_params(p)

    # Ensure metrics directory exists
    metrics_dir = p["metrics_dir"]
    os.makedirs(metrics_dir, exist_ok=True)

    # Prepare the data module
    dm = LineListingDataModule(
        seq_len=p["seq_len"],
        output_size=p["output_size"],
        batch_size=p["batch_size"],
        num_workers=p["num_workers"],
    )

    # Load and preprocess data
    X, y = dm.load_and_preprocess_data()
    logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    # Define a unique run name using the current timestamp
    run_name = f"test_run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    # Start MLflow experiment
    with mlflow.start_run(run_name=run_name):
        # Log hyperparameters
        mlflow.log_params({
            "learning_rate": p["learning_rate"],
            "batch_size": p["batch_size"],
            "epochs": p["max_epochs"],
            "model_type": "LSTM",
            "seq_len": p["seq_len"],
            "output_size": p["output_size"],
            "hidden_size": p["hidden_size"],
            "num_layers": p["num_layers"],
            "dropout": p["dropout"],
        })

        # Initialize the model
        model = LSTMRegressor(
            n_features=p["n_features"],
            hidden_size=p["hidden_size"],
            criterion=p["criterion"],
            num_layers=p["num_layers"],
            dropout=p["dropout"],
            learning_rate=p["learning_rate"],
            batch_size=p["batch_size"],  # Pass batch_size to the model
            output_size=p["output_size"],
        )

        # Log model summary
        logger.info(summary(model, input_size=(p["batch_size"], p["seq_len"], p["n_features"]), col_names=["input_size", "output_size", "num_params"]))
        # print(summary(model, input_size=(p["batch_size"], p["seq_len"], p["n_features"]), col_names=["input_size", "output_size", "num_params"]))

        # Set up the trainer
        trainer = Trainer(
            accelerator="auto",
            max_epochs=p["max_epochs"],
            logger=[csv_logger],
            callbacks=configure_callbacks(metrics_dir),
            log_every_n_steps=1,
            enable_progress_bar=False,
            enable_model_summary=True,
        )

        # Log device information
        device = trainer.strategy.root_device
        logger.info(f"Using device: {device.type} ({device})")

        # Train the model
        trainer.fit(model, dm)

        # Log training and validation metrics
        train_loss = trainer.callback_metrics.get("train_loss", 0.0)
        val_loss = trainer.callback_metrics.get("val_loss", 0.0)
        mlflow.log_metrics({
            "train_loss": train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss,
            "val_loss": val_loss.item() if isinstance(val_loss, torch.Tensor) else val_loss
        })

        # Log the trained model
        mlflow.pytorch.log_model(model, "model")

        # Initialize results dictionary for CSV logging
        results = {
            "experiment_id": mlflow.active_run().info.run_id,
            "learning_rate": p["learning_rate"],
            "batch_size": p["batch_size"],
            "epochs": p["max_epochs"],
            "train_loss": train_loss,
            "val_loss": val_loss,
        }

        # Test the model (if enabled)
        if p.get("run_test", True):
            test_results = trainer.test(model, datamodule=dm)

            # Log test metrics if available
            if test_results:
                for metric_name, metric_value in test_results[0].items():
                    mlflow.log_metric(f"test_{metric_name}", metric_value.item() if isinstance(metric_value, torch.Tensor) else metric_value)
                    results[f"test_{metric_name}"] = metric_value.item() if isinstance(metric_value, torch.Tensor) else metric_value

if __name__ == "__main__":
    main()
