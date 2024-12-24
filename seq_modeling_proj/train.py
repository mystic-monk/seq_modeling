# train.py
from data_set import LineListingDataModule
from lightning import Trainer
from base_model import LSTMRegressor
from config import p, tensorboard_logger, validate_params, csv_logger 
from callbacks import PrintingCallback, early_stop_callback, checkpoint_callback, CSVLoggerCallback


def main():
    """
    Main script for training and testing the LSTM model on time series data.
    Parameters are sourced from `config.py`.
    """

    validate_params(p)


    # set the data module
    dm = LineListingDataModule(
        seq_len=p["seq_len"],  # type: ignore
        batch_size=p["batch_size"],  # type: ignore
        num_workers=p["num_workers"],  # type: ignore
    )

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
