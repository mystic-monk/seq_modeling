# train.py
import os
from data_set import LineListingDataModule
from lightning import Trainer
from config import p
from callbacks import PrintingCallback, early_stop_callback, checkpoint_callback, CSVLoggerCallback
import warnings
from torchinfo import summary
from rich.console import Console
from rich.table import Table
from model import Seq2Seq, Encoder, Decoder
import torch
from expanding_window import ExpandingWindow
import numpy as np


def print_config(params):
    """
    Print the current configuration parameters in a table format.
    
    Args:
        params (dict): Dictionary of configuration parameters.
    
    Returns:
        None
    
    Raises:
        TypeError: If `params` is not a dictionary.
    """
    table = Table(title="Configuration Parameters", title_style="bold cyan")
    table.add_column("Parameter", style="bold yellow")
    table.add_column("Value", style="bold white")
    
    for key, value in params.items():
        table.add_row(key, str(value))

    table.add_row("initial", str(params.get("initial", "Not Set")))
    table.add_row("horizon", str(params.get("horizon", "Not Set")))
    table.add_row("period", str(params.get("period", "Not Set")))

    console.print(table)
    console.print("#" * 42, style="cyan")


def configure_callbacks(metrics_dir):
    """
    Configure callbacks for the trainer.

    Args:
        metrics_dir (str): Directory to store metrics and logs.

    Returns:
        list: Configured callbacks.
    """

    return [
        PrintingCallback(verbose=True),
        early_stop_callback,
        checkpoint_callback,
        CSVLoggerCallback(
            train_log_file=os.path.join(metrics_dir, "train_metrics.csv"),
            val_log_file=os.path.join(metrics_dir, "val_metrics.csv"),
            test_log_file=os.path.join(metrics_dir, "test_metrics.csv"),
            train_predictions_file=os.path.join(metrics_dir, "train_predictions.csv"),
            val_predictions_file=os.path.join(metrics_dir, "val_predictions.csv"),
            test_predictions_file=os.path.join(metrics_dir, "test_predictions.csv"),
        ),
    ]


def generate_walk_forward_splits(data, params):
    """
    Generate walk-forward splits using the ExpandingWindow class.

    Args:
        data (array-like): The input data to split.
        params (dict): Configuration parameters containing initial, horizon, and period.

    Returns:
        list: A list of tuples (train_indices, test_indices).
    """
    ew = ExpandingWindow(
        initial=params["initial"],
        horizon=params["horizon"],
        period=params["period"]
    )
    return ew.split(data)


def create_model(params):
    """
    Create the Seq2Seq model.

    Args:
        params (dict): Configuration parameters.

    Returns:
        Seq2Seq: The initialized model.
    """
    encoder = Encoder(
        input_size=params["n_features"],
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        rnn_type=params["rnn_type"],
        dropout=params["dropout"],
    )
    decoder = Decoder(
        output_size=params["output_size"],
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        rnn_type=params["rnn_type"],
        dropout=params["dropout"],
    )
    # return Seq2Seq(encoder, decoder, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return Seq2Seq(encoder, decoder)

def main():
    """
    Main script for training and testing the Seq2Seq model on time series data.
    """
    console.print("#" * 42, style="cyan")
    console.print("Starting Training ...", style="bold green")
    console.print("#" * 42, style="cyan")

    # Log configuration parameters
    print_config(p)

    # Prepare the data module
    dm = LineListingDataModule(
        seq_len=p["seq_len"],
        output_size=p["output_size"],
        batch_size=p["batch_size"],
        num_workers=p["num_workers"],
        initial=p["initial"],  
        horizon=p["horizon"],  
        period=p["period"],   
    )

    # Exception handling for data module setup
    try:
        dm.prepare_data()
        # Mark as already prepared and set up to prevent Trainer from re-calling them
        dm.has_prepared_data = True        
        
    except Exception as e:
        console.print(f"[bold red]Error during data preparation: {e}[/bold red]")
        return


    # Generate walk-forward splits
    splits = generate_walk_forward_splits(data=np.arange(len(dm.X_train_val)), params=p)

    for i, (train_indices, val_indices) in enumerate(splits):
        console.print(f"Processing Split {i+1}/{len(splits)}", style="bold cyan")

        # Setup data for the current split      
        dm.setup_split(train_indices, val_indices)
        dm.setup()
        dm.has_setup = True
        # Create the model
        model = create_model(p)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        # print(model)


        # Log model summary
        console.print("Model Summary:", style="bold magenta")
        # console.print(summary(model, input_size=(p["batch_size"], p["seq_len"], p["n_features"]), 
        #                     col_names=["input_size", "output_size", "num_params"], verbose=0))

        # Set up the trainer
        trainer = Trainer(
            max_epochs=p["max_epochs"],
            callbacks=configure_callbacks(p["metrics_dir"]),
            log_every_n_steps=p.get("log_every_n_steps", 1),
            check_val_every_n_epoch=p.get("check_val_every_n_epoch", 1),
            enable_progress_bar=False,
            enable_model_summary=True,
        )


        # Log device information
        device = trainer.strategy.root_device
        console.print(f"Using device: [bold green]{device.type} ({device})[/bold green]")


        # Train the model
        try:
            console.print("Training has started!", style="bold cyan")
            trainer.fit(model, dm)
        except Exception as e:
            console.print(f"[bold red]Error during training: {e}[/bold red]")
            return


        # Test the model (if enabled)
        # if p.get("run_test", True):
        #     console.print("Testing the model...", style="bold blue")
        #     try:
        #         test_results = trainer.test(model, dm, verbose=False)
        #         table = Table(title="Test Results", title_style="bold green")
        #         table.add_column("Metric", style="cyan", justify="left")
        #         table.add_column("Value", style="yellow", justify="right")

        #         if len(test_results) > 0:
        #             for metric, value in test_results[0].items():
        #                 table.add_row(metric, f"{value:.4f}")
        #         else:
        #             console.print("No test results found.", style="bold red")
        #         console.print(table)
        #     except Exception as e:
        #         console.print(f"[bold red]Error during testing: {e}[/bold red]")
        
 
if __name__ == "__main__":
    # Initialize rich console
    console = Console()

    # Suppress specific warnings for cleaner output
    warnings.filterwarnings(
        "ignore",
        message="The '.*dataloader.*' does not have many workers.*",
        category=UserWarning,
    )
    main()

