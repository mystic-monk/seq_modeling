# train.py
from data_set import LineListingDataModule
from lightning import Trainer
from model import LSTMRegressor
import optuna
from config import p, tensorboard_logger, validate_params, csv_logger 
from callbacks import PrintingCallback, early_stop_callback, checkpoint_callback, CSVLoggerCallback

# Function to load the best hyperparameters from the Optuna study
def get_best_hyperparameters(study):
    best_trial = study.best_trial
    best_params = best_trial.params
    return best_params

def main():
    """
    Main script for training and testing the LSTM model on time series data.
    Parameters are sourced from `config.py`.
    """

    validate_params(p)

    # Load the Optuna study
    study = optuna.load_study(study_name="lstm_study", 
                              storage="sqlite:///optuna_lstm.db"
                              )

    # Get the best hyperparameters
    best_params = get_best_hyperparameters(study)
    if best_params["batch_size"] < 2:
        raise ValueError("Batch size must be at least 2 for R² computation.")
    print(f"Best hyperparameters: {best_params}")

    # set the data module
    dm = LineListingDataModule(
        seq_len=p["seq_len"],  # type: ignore
        batch_size=best_params["batch_size"],  # type: ignore
        num_workers=p["num_workers"],  # type: ignore
    )

    # Build the model
    model = LSTMRegressor(
        n_features=p["n_features"],
        hidden_size=best_params["hidden_size"],
        criterion=p["criterion"],
        num_layers=best_params["num_layers"],
        dropout=best_params["dropout"],
        learning_rate=best_params["learning_rate"],
        output_size=p["output_size"],
    )

    trainer = Trainer(
        accelerator="auto",
        max_epochs=best_params["max_epochs"],
        logger=[tensorboard_logger, csv_logger],
        callbacks=[
            PrintingCallback(), 
            early_stop_callback, 
            checkpoint_callback, 
            CSVLoggerCallback(log_file="training_metrics.csv")
        ],
        benchmark=True,
        log_every_n_steps=5,
    )

    # Log dataset sizes
    # print(f"Dataset sizes: Train={len(dm.train_dataloader())}, "
    #       f"Val={len(dm.val_dataloader())}, Test={len(dm.test_dataloader())}")

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
