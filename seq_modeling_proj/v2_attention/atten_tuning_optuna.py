import optuna
from lightning import Trainer
from data_set import LineListingDataModule
from atten_model import HybridModel  # Import the hybrid model
from config import p  # Config as a base
from lightning.pytorch.callbacks import EarlyStopping

def objective(trial):
    """
    Objective function for Optuna to minimize.
    Args:
        trial (optuna.trial.Trial): An Optuna trial object.
    Returns:
        float: Validation loss for the trial.
    """

    # Suggest hyperparameters for LSTM and Attention
    hidden_size = trial.suggest_int("hidden_size", 16, 128, step=16)
    num_layers = trial.suggest_int("num_layers", 1, 4)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    max_epochs = trial.suggest_int("max_epochs", 10, 50, step=5)

    # Attention-specific hyperparameters
    attention_heads = trial.suggest_int("attention_heads", 2, 8, step=2)
    attention_layers = trial.suggest_int("attention_layers", 1, 3)

    # Update the data module with the trial's batch size
    dm = LineListingDataModule(
        seq_len=p["seq_len"],
        batch_size=batch_size,
        num_workers=p["num_workers"]
    )

    # Build the model with trial's hyperparameters
    model = HybridModel(
        n_features=p["n_features"],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        learning_rate=learning_rate,
        criterion=p["criterion"],
        output_size=p["output_size"],
        attention_heads=attention_heads,  # Pass attention heads
        attention_layers=attention_layers,  # Pass attention layers
    )

    # Callbacks
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=3, mode="min")

    # Set up the PyTorch Lightning Trainer
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        logger=False,  # Disable logging to speed up tuning
        enable_checkpointing=False,  # No need for checkpoints during tuning
        callbacks=[early_stop_callback],
    )

    # Train the model
    trainer.fit(model, dm)

    # Retrieve validation loss
    val_loss = trainer.callback_metrics.get("val_loss")
    if val_loss is None:
        print("Warning: Validation loss not found, returning a large loss.")
        return float("inf")
    
    # Manual pruning logic
    trial.report(val_loss.item(), step=max_epochs)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return val_loss.item()


if __name__ == "__main__":
    # Create or load the study
    study = optuna.create_study(
        direction="minimize",
        study_name="hybrid_model_study",  # Updated study name
        storage="sqlite:///optuna_hybrid.db",  # Updated storage file
        load_if_exists=True,
    )

    # Optimize
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    # Print the best trial
    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
