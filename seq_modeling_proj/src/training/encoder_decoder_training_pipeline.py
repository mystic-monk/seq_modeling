# Standard library imports
import pandas as pd 
# Third-party imports
import torch
# import torch.nn as nn
import time
from pathlib import Path
from datetime import datetime
import psutil
# Local application imports
from utils.data.data_utils import train_val_data_path

from models.lstm import LSTMRegressor
import os 
from utils.config.config_setup import config, logger

from data_pipeline.dataloader.data_loaders import train_dataloader, val_dataloader
from data_pipeline.preprocessing.data_preprocessing import scale_data, load_and_preprocess_data
from data_pipeline.preprocessing.expanding_window import ExpandingWindow
from utils.metrics import r2_score, mse_loss

def train_one_epoch(model, train_loader, optimizer, device, teacher_forcing_ratio=0.5):
    """Runs one epoch of training for the encoder-decoder model."""
    model.train()
    train_loss = 0.0
    gradient_norms = []

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        y_pred = model(x, y, teacher_forcing_ratio)  # Pass y for teacher forcing
        loss = model.criterion(y_pred, y)
        loss.backward()
        
        # Gradient monitoring
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        gradient_norms.append(total_norm ** 0.5)

        optimizer.step()
        train_loss += loss.item()

    return {
        "train_loss": train_loss / len(train_loader),
        "avg_grad_norm": sum(gradient_norms) / len(gradient_norms),
        "max_grad_norm": max(gradient_norms)
    }


def validate(model, val_loader, device):
    """Runs validation for the encoder-decoder model."""
    model.eval()
    val_loss = 0.0
    y_preds, y_trues = [], []

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x, None)  # No teacher forcing during validation
            loss = model.criterion(y_pred, y)

            val_loss += loss.item()
            y_preds.append(y_pred)
            y_trues.append(y)

    y_preds = torch.cat(y_preds, dim=0)
    y_trues = torch.cat(y_trues, dim=0)
    val_loss /= len(val_loader)

    return {
        "val_loss": val_loss,
        "mse": mse_loss(y_preds, y_trues).item(),
        "rmse": torch.sqrt(mse_loss(y_preds, y_trues)).item(),
        "y_preds": y_preds.cpu().numpy(),
        "y_trues": y_trues.cpu().numpy()
    }

def save_artifacts(metrics_dir, split_idx, epoch, results, config):
    """Save all training artifacts to structured CSV files"""
    # Save metrics
    metrics_path = metrics_dir / f"split_{split_idx:02}_metrics.csv"
    pd.DataFrame([results]).to_csv(metrics_path, mode='a', 
                                 header=not os.path.exists(metrics_path),
                                 index=False)
    
    # Save predictions
    preds_df = pd.DataFrame({
        "y_true": results["y_trues"].flatten(),
        "y_pred": results["y_preds"].flatten()
    })
    preds_path = metrics_dir / f"split_{split_idx:02}_epoch_{epoch:03}_predictions.csv"
    preds_df.to_csv(preds_path, index=False)
    

def run_training(metrics_dir: str, checkpoints_dir, model:LSTMRegressor, train_loader, val_loader, 
                 num_epochs, device,  split_idx, scaler, patience=50):
    """Enhanced training loop with comprehensive tracking"""
    
    # Move model to the specified device
    model.to(device)
    optim_config = model.configure_optimizers()
    optimizer = optim_config["optimizer"]
    scheduler = optim_config["scheduler"]

    epochs_no_improve = 0 
    best_val_loss = float("inf")

    history = []
    logger.info("[bold magenta]🚀 Starting training...[/bold magenta]")

    # Training loop
    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Training phase
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        
        # Validation phase
        val_results = validate(model, val_loader, device)
  
        # Learning rate tracking
        current_lr = optimizer.param_groups[0]['lr']
        
        # Memory monitoring
        memory_used = torch.cuda.max_memory_allocated(device) / (1024 ** 3) if torch.cuda.is_available() else 0

        # Build results dictionary
        epoch_results = {
            "split_idx": split_idx,
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "epoch_duration": time.time() - epoch_start,
            "learning_rate": current_lr,
            "train_loss": train_metrics["train_loss"],
            "val_loss": val_results["val_loss"],
            # "r2_score": val_results["r2"],
            "mse": val_results["mse"],
            "rmse": val_results["rmse"],
            "avg_grad_norm": train_metrics["avg_grad_norm"],
            "max_grad_norm": train_metrics["max_grad_norm"],
            "gpu_memory_gb": memory_used,
            "system_memory_percent": psutil.virtual_memory().percent,
            "y_preds": val_results["y_preds"],
            "y_trues": val_results["y_trues"]
        }
        
        # Add model hyperparameters
        epoch_results.update({
            "hidden_size": model.hidden_size,
            "num_layers": model.num_layers,
            "dropout": model.dropout,
            "batch_size": config["batch_size"]
        })

        history.append(epoch_results)
        save_artifacts(metrics_dir, split_idx, epoch, epoch_results, config)


        # Check for improvement
        if val_results["val_loss"] < best_val_loss:
            best_val_loss = val_results["val_loss"]
            epochs_no_improve = 0 

            # Update the filename with the current epoch and validation loss
            filename = (
                f"ew_split_{split_idx:02}"
                f"_epoch_{epoch+1:03}"
                f"_model_lstm_lr_{current_lr:.6f}_loss_{val_results['val_loss']:.4f}"
                f"_batch_{config['batch_size']}_layers_{config['num_layers']}_dropout_{config['dropout']}"
                f".pth")

            torch.save({
                "model_state": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "h0": model.h0,
                "c0": model.c0,
                "X_scaler": scaler,
                "epoch": epoch,
                "split": split_idx,
                "metrics": epoch_results
                }, checkpoints_dir / f"{filename}")
        else:
            epochs_no_improve += 1
            

        logger.info(
            f"[bold red]📊 Sp: {split_idx+1} [/bold red]| [bold green] Ep: {epoch+1} [/bold green] | "
            f"[bold blue] TLoss: {epoch_results['train_loss']:.4f} [/bold blue] | "
            f"[bold cyan] VLoss: {epoch_results['val_loss']:.4f} [/bold cyan] | "
            f"[bold green] BLoss: {best_val_loss:.4f} [/bold green] | "
            # f"R2: {val_results['r2']:.4f} | "
            f"LR: {current_lr:.2e}")
        
        # Step the scheduler
        scheduler.step(val_results["val_loss"])

        # Stop training if patience is exceeded
        if epochs_no_improve >= patience:
            logger.debug(f"[bold red]🛑 Early stopping triggered. Training stopped at epoch {epoch}.[/bold red]")
            break

    return history


if __name__ == "__main__":
    
    start_time = time.time()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoints_dir = config['experiment_dirs']['checkpoints_dir']
    metrics_dir = config['experiment_dirs']['metrics_dir']

    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent
    metrics_dir_path = project_root / Path(metrics_dir).relative_to("/") if metrics_dir.startswith("/") else Path(metrics_dir)

    checkpoints_dir_path = project_root / Path(checkpoints_dir).relative_to("/") if checkpoints_dir.startswith("/") else Path(checkpoints_dir)

    experiment_name = f"{config['experiment']['experiment_name']}"

    # Load train, validation, and test data
    raw_data = train_val_data_path(config)
    X_df, y_df = load_and_preprocess_data(raw_data, config["data_columns"], config["seq_len"], logger)

    X_raw = X_df.values
    y_raw = y_df.values

    # Initialize Expanding Window : horizon is the validation set in the split
    exp_window = ExpandingWindow(initial=30, horizon=28, period=28) 
    splits = exp_window.split(X_raw)  

    full_history = []

    # Loop through expanding window splits
    for split_idx, (train_idx, val_idx) in enumerate(splits):
        logger.info(f"[bold green]Processing Split {split_idx+1}/{len(splits)}[/bold green]")

        # Scale data per split
        X_train, y_train, scaler_train = scale_data(X_raw[train_idx], y_raw[train_idx])
        X_val, y_val, _ = scale_data(X_raw[val_idx], y_raw[val_idx])

        # Initialize the model
        model = LSTMRegressor(
            n_features=config["n_features"],
            hidden_size=config["hidden_size"],
            criterion=config["criterion"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            learning_rate=config["learning_rate"],
            batch_size=config["batch_size"],  
            output_size=config["output_size"],
        )

        # Create DataLoaders for this split
        train_loader = train_dataloader(X_train, y_train, config["seq_len"], 
                                        config["output_size"], config["batch_size"], 
                                        config["num_workers"], logger)
        val_loader = val_dataloader(X_val, y_val, config["seq_len"], 
                                    config["output_size"], config["batch_size"], 
                                    config["num_workers"], logger)

        # Train the model
        split_history = run_training(metrics_dir_path, checkpoints_dir_path, model, train_loader, 
                                     val_loader, num_epochs=200, 
                                     device=device, split_idx=split_idx, scaler=scaler_train)

        full_history.extend(split_history)
        torch.cuda.empty_cache()

    # End the timer and calculate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Save complete history
    full_history_path = metrics_dir_path / "full_training_history.csv"
    pd.DataFrame(full_history).to_csv(full_history_path, index=False)

    logger.info(f"[bold green]Training completed in {elapsed_time / 60:.2f} minutes[/bold green]")
