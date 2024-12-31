# train.py
import os
from data_set import LineListingDataModule
from lightning import Trainer
from model import LSTMRegressor
from config import p, tensorboard_logger, validate_params, csv_logger 
from callbacks import PrintingCallback, early_stop_callback, checkpoint_callback, CSVLoggerCallback
from sklearn.model_selection import KFold

import warnings

warnings.filterwarnings(
    "ignore",
    message="The '.*dataloader.*' does not have many workers.*",
    category=UserWarning,
)

def main():
    """
    Main script for training and testing the LSTM model on time series data.
    Parameters are sourced from `config.py`.
    """

    validate_params(p)

    # Ensure the 'metrics' directory exists
    metrics_dir = p["metrics_dir"]
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)

    # set the data module
    dm = LineListingDataModule(
        seq_len=p["seq_len"],  # type: ignore
        batch_size=p["batch_size"],  # type: ignore
        num_workers=p["num_workers"],  # type: ignore
    )

    # Get the dataset for KFold splitting
    X, y = dm.load_and_preprocess_data()

    # Initialize KFold
    kf = KFold(n_splits=5)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Fold {fold + 1}")

        # Split the dataset
        dm.setup_fold(train_idx, val_idx)

        # Build the model
        model = LSTMRegressor(
            n_features=p["n_features"],
            hidden_size=p["hidden_size"],
            criterion=p["criterion"],
            num_layers=p["num_layers"],
            dropout=p["dropout"],
            learning_rate=p["learning_rate"],
            output_size=p["output_size"],
        )

        trainer = Trainer(
            accelerator="auto",
            max_epochs=p["max_epochs"],
            logger=[tensorboard_logger, csv_logger],
            callbacks=[
                PrintingCallback(), 
                early_stop_callback, 
                checkpoint_callback, 
                CSVLoggerCallback(log_file=os.path.join(metrics_dir,f"training_metrics_fold_{fold + 1}.csv"))
            ],
            benchmark=True,
            log_every_n_steps=5,
            enable_progress_bar=False,  # Disable the progress bar
        )

        # Log device being used
        print(f"Using device: {trainer.strategy.root_device.type} "
              f"({trainer.strategy.root_device})")

        trainer.fit(model, dm)

        # Test the model (optional)
        if p.get("run_test", True):
            trainer.test(model, datamodule=dm)

if __name__ == "__main__":
    main()
    # tensorboard --logdir=runs
