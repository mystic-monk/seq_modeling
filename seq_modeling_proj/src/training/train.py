# Standard library imports
import os

# Third-party imports
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch

# Local application imports
from utils.logging_setup import get_logger, train_val_data_path
from models.lstm import LSTMRegressor
# from utils.config_loader import load_config
from utils.config_setup import p
from data_pipeline.dataloader.data_loaders import train_dataloader, val_dataloader
from data_pipeline.preprocessing.data_preprocessing import scale_data, load_and_preprocess_data
from data_pipeline.preprocessing.expanding_window import ExpandingWindow
from utils.mlflow_setup import setup_mlflow

# Get the logger from the centralized setup
logger = get_logger()

# p = load_config()
# Initialize MLflow
setup_mlflow()

model_dir = p["experiment_dirs"]["checkpoints_dir"]

def r2_score(y_pred, y_true):
    ss_total = torch.sum((y_true - torch.mean(y_true, dim=0)) ** 2, dim=0)
    ss_residual = torch.sum((y_true - y_pred) ** 2, dim=0)
    r2_per_feature = 1 - (ss_residual / (ss_total + 1e-8))  # Avoid division by zero
    return torch.mean(r2_per_feature)  # Average across all output features

def mse_loss(y_pred, y_true):
    """
    Compute Mean Squared Error (MSE) loss manually for vector outputs.
    
    Args:
        y_pred (torch.Tensor): Predicted values of shape [batch_size, output_size]
        y_true (torch.Tensor): Ground truth values of shape [batch_size, output_size]
    
    Returns:
        torch.Tensor: Scalar MSE loss
    """
    return torch.mean((y_pred - y_true) ** 2)  # Computes MSE over all elements


def run_training(model, train_loader, val_loader, num_epochs, device,  split_idx, best_val_loss = float("inf"), patience=50):
    """
    Train the LSTM model while logging metrics and saving the best model with MLflow.
    """
    # Move model to the specified device
    model.to(device)
    if isinstance(model.criterion, str):
        model.criterion = getattr(nn, model.criterion)()
    
    # Configure optimizer and scheduler
    optimizer, scheduler = model.configure_optimizers().values()


    logger.info("[bold magenta]🚀 Starting training...[/bold magenta]")

    epochs_no_improve = 0 
    best_model_weights = None
    prev_best_weights = model.state_dict()


    # 🎯 Start MLflow experiment for this split
    with mlflow.start_run(run_name=f"Expanding_Window_Split_{split_idx}") as run:
        mlflow.log_params({
            "learning_rate": optimizer.param_groups[0]['lr'],
            "batch_size": train_loader.batch_size,
            "num_layers": model.num_layers,
            "hidden_size": model.hidden_size,
            "dropout": model.dropout,
            "criterion": str(model.criterion),
            "sequence_length": train_loader.dataset.seq_len
                               
        })

        # Training loop
        for epoch in range(num_epochs):
            logger.info(f"[bold yellow]Epoch {epoch+1}/{num_epochs}[/bold yellow]")
            model.train()
            train_loss = 0.0

            for batch_idx, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)

                # Forward pass
                y_pred = model(x)
                loss = model.criterion(y_pred, y)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
    

            train_loss /= len(train_loader)
            # print(f"Training Loss: {train_loss:.4f}", end="\r")

            # Validation loop
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    y_pred = model(x)
                    loss = model.criterion(y_pred, y)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            # 🎯 Log training & validation loss
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0 
                best_model_weights = model.state_dict()

                # Save the best model weights
                logger.info(f"Best model weights saved at epoch {epoch+1} with validation loss {val_loss:.4f}")
                # Create dynamic save path
                current_lr = optimizer.param_groups[0]['lr']
                # Update the filename with the current epoch and validation loss
                filename = (
                    f"ew_split_{split_idx:02}"
                    f"_epoch_{epoch+1:03}"
                    f"_model_lstm_lr_{current_lr:.6f}_loss_{val_loss:.4f}"
                    f"_batch_{p['batch_size']}_layers_{p['num_layers']}_dropout_{p['dropout']}"
                    f".pth")

                torch.save(model.state_dict(), os.path.join(model_dir, filename))

                # 🎯 Log model to MLflow
                mlflow.pytorch.log_model(model, f"best_model_split_{split_idx}")

                logger.info(f"[bold green]✅ New best model saved at epoch {epoch+1} with validation loss {val_loss:.4f}[/bold green]")
            else:
                epochs_no_improve += 1
                logger.info(f"No improvement for {epochs_no_improve}/{patience} epochs.")


            logger.info(f"[bold cyan]📊 Split: {split_idx+1} | Epoch: {epoch+1} | T Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Best Loss: {best_val_loss:.4f}[/bold cyan]")
            
            # Step the scheduler
            scheduler.step(val_loss)

            # Stop training if patience is exceeded
            if epochs_no_improve >= patience:
            
                logger.debug("[bold red]🛑 Early stopping triggered. Training stopped.[/bold red]")
                break



    # Load best model before returning (only if it's updated)
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
    else:
        model.load_state_dict(prev_best_weights)
        logger.warning("No model improvements found during training.")

    return model, best_model_weights, best_val_loss

if __name__ == "__main__":
    

    # Load train, validation, and test data
    train_val_data = train_val_data_path()

    X_train_val, y_train_val = load_and_preprocess_data(train_val_data, p["data_columns"], p["seq_len"])

    # ✅ Scale the data BEFORE passing it to DataLoaders
    X_train_val, y_train_val = scale_data(X_train_val, y_train_val)


    # Initialize Expanding Window
    exp_window = ExpandingWindow(initial=30, horizon=28, period=28) 
    splits = exp_window.split(X_train_val)

        # Initialize the model
    model = LSTMRegressor(
        n_features=p["n_features"],
        hidden_size=p["hidden_size"],
        criterion=p["criterion"],
        num_layers=p["num_layers"],
        dropout=p["dropout"],
        learning_rate=p["learning_rate"],
        batch_size=p["batch_size"],  
        output_size=p["output_size"],

    )

    best_val_loss = float("inf")
    best_weights = None
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Loop through expanding window splits

    for i, (train_idx, val_idx) in enumerate(splits):
        logger.info(f"[bold green]Running Expanding Window {i+1}/{len(splits)}[/bold green]")

        # Create train and validation sets
        X_train, y_train = X_train_val[train_idx], y_train_val[train_idx]
        X_val, y_val = X_train_val[val_idx], y_train_val[val_idx]
        logger.info(f"[bold blue]Train shape: {X_train.shape}, {y_train.shape}, Val shape: {X_val.shape}, {y_val.shape}[/bold blue]")


        # Create DataLoaders for this split
        train_loader = train_dataloader(X_train, y_train, p["seq_len"], p["output_size"], p["batch_size"], p["num_workers"])
        val_loader = val_dataloader(X_val, y_val, p["seq_len"], p["output_size"], p["batch_size"], p["num_workers"])

        # Load previous model weights if available
        if best_weights is not None:
            model.load_state_dict(best_weights)
            logger.info(f"[bold green]🔄 Loaded previous model weights for continued training.[/bold green]")


        # Train the model
        model, best_weights, best_val_loss = run_training(model, train_loader, val_loader, num_epochs=200, device=device, split_idx=i, best_val_loss=best_val_loss )
        # print(f"Best validation loss: {best_val_loss:.4f}")
        # print(f"Best model weights: {best_weights}")
